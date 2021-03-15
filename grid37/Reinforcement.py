import time 
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class QLearner():
    """
        Classe principale d'apprentissage par renforcement par Q learning. Permet d'effectuer
        cette optimisation sur l'environnement voulu
        
        Args: - env: environnement choisi
              - algo: choix du type d'algorithme 'simple' Q learning ou 'double' Q learning
              - nb_action: nombre d'actions utilisées dans l'environnement
    """
    
    def __init__(self, env, algo = "simple", nb_action = 5):
        self.algo = algo
        self.env = env
        self.nb_action = nb_action
        self.Q = None
        self.max_horizon = -1
        self.case = env.mg.architecture['genset'] # Défini le type d'environnement


    def reset(self, testing=False, sampling_args = None):
        """ Wrapper de la méthode reset d'une microgrid permettant de gérer l'augmentation
            de la dimension de Q
            
            Args: - testing: active ou non le mode test sur l'environnement
                  - sampling_args: argument de la méthode reset de l'environnement
                  
            Output: retourne le résultat de la méthode transition
        """
        self.env.reset(testing, sampling_args)
        return self.transition()

    
    def transition(self, env=None):
        """ Wrapper de la méthode transition d'une microgrid permettant de
            créer des states custom (peut prendre un env différent)
            
            Args: - env: environnement sur lequel effectué le step (peut être différent de 
                          l'environnement utilisé pour l'init)
            
            Input: env -> (optionnel) permet de réaliser la transition sur un autre
            environnement que celui stocké dans l'objet QLearner (utile pour le test sur 
            un autre environnement)
            
            Output: retourne le nouvel état dans la nouvelle dimension
        """
        if env is None:
            state = self.env.transition()
        else:
            state = env.transition()
        ### ICI AGIR SUR LES ETATS CUSTOMS
        
        # Heure pleine, heure creuse, heure intermédiaire
        hour = self.env.mg.get_updated_values()["hour"]
        if hour in range(0,8) or hour in range(21,24):
            # Heure creuse
            peak_hours = 0
        elif hour in range(8,12) or hour in range(18,21):
            # Heure intermédiaire
            peak_hours = 1
        else:
            # Heure pleine
            peak_hours = 2
            
        # Production supérieur à la demande sur les heures d'ensoleillement
        forecast_load = self.env.mg.forecast_load()
        forecast_pv = self.env.mg.forecast_pv()
        forecast_diff = (forecast_load - forecast_pv)[8:21]
        overproduction = 0 if sum(forecast_diff) > 0 else 1

        # Espace spécifique à la 3ème microgrid
        if self.case == 1:
            grid = self.env.mg.get_updated_values()['grid_status']
            return (grid, overproduction, peak_hours, *state)
        return (overproduction, peak_hours, *state)


    def step(self, action, env=None):
        """Wrapper de la méthode step d'une microgrid permettant de modifier la fonction de reward
        
            Args: - action: action choisie
                  - env: environnement sur lequel effectué le step (peut être différent de 
                          l'environnement utilisé pour l'init)
                          
            Output: - state: nouvel état calculé
                    - reward: reward calculé pour l'action donnée
                    - done: indicateur d'état de la microgrid
                    - info: permet de rajouter des infos (non utilisé)
        """
        if env is None:
            _, reward, done, info = self.env.step(action)
            state = self.transition()
        else:
            _, reward, done, info = env.step(action)
            state = self.transition(env)
        
        ### ICI AGIR SUR LE REWARD
        if action == 0:
            reward += 0
        return state, reward, done, info


    def update_Q(self, Q, step, horizon, reward, action, state, future_state, 
                 alpha, epsilon, gamma, algo):
        """ Simple Q Learning: fonction de mise à jour de Q
        
            Args: - Q: matrice Q à mettre à jour
                  - step: indice de temps sur lequel on travaille
                  - horizon: horizon maximale
                  - reward: reward obtenu par l'action précédente
                  - action: action choisie
                  - state: état choisi
                  - future_state: état futur choisi
                  - alpha: learning rate
                  - epsilon: paramètre du epsilon greedy
                  - gamma: paramètre gamma
                  - algo: type d'algorithme utilisé 'q_learning' ou 'sarsa'
            
            Output: - la nouvelle matrice Q
                    - la nouvelle action
        """
        if algo == "q_learning":
            future_action = max_dict(Q[future_state])[0] # On choisi l'action avec le poids le plus fort
            if step == horizon-1:
                Q[state][action] += alpha*(reward - Q[state][action])
            else:
                target = reward + gamma*Q[future_state][future_action]
                Q[state][action] = (1-alpha) * Q[state][action] + alpha * target
        elif algo == "sarsa":
            future_action = max_dict(Q[future_state])[0] # On choisi l'action avec le poids le plus fort
            if step == horizon-1:
                Q[state][action] += alpha*(reward - Q[state][action])
            else:
                a, randomm = espilon_decreasing_greedy(action, epsilon, self.nb_action)
                target = reward + gamma*Q[future_state][a]
                Q[state][action] = (1-alpha) * Q[state][action] + alpha * target
        else:
            print(f"Algo {algo} not implemented: using q_learning")
            Q, future_action = self.update_Q(Q, step, horizon, reward, action, state, future_state, 
                 alpha, epsilon, gamma, "q_learning")
        return Q, future_action
    

    def update_doubleQ(self, Q, step, horizon, reward, action, state, future_state, 
                       alpha, epsilon, gamma, algo):
        """ Double Q Learning: fonction de mise à jour de Q = (QA, QB) 
        
            Args: - Q: matrice Q à mettre à jour
                  - step: indice de temps sur lequel on travaille
                  - horizon: horizon maximale
                  - reward: reward obtenu par l'action précédente
                  - action: action choisie
                  - state: état choisi
                  - future_state: état futur choisi
                  - alpha: learning rate
                  - epsilon: paramètre du epsilon greedy
                  - gamma: paramètre gamma
                  - algo: type d'algorithme utilisé 'q_learning' ou 'sarsa'
            
            Output: - la nouvelle matrice Q = (QA, QB)
                    - la nouvelle action
        """
        if algo == "q_learning":
            future_action = max_dict2(Q[0][future_state], Q[1][future_state])[0]
            p = np.random.random()
            idx = 0 if p < 0.5 else 1 # On choisi quelle matrice va être mise à jour
            if step == horizon-1:            
                Q[idx][state][action] += alpha*(reward - Q[idx][state][action])
            else:
                target = reward + gamma * Q[1-idx][future_state][max_dict(Q[idx][future_state])[0]]
                Q[idx][state][action] = (1-alpha) * Q[idx][state][action] + alpha * target
        elif algo == "sarsa":
            future_action = max_dict2(Q[0][future_state], Q[1][future_state])[0]
            p = np.random.random()
            idx = 0 if p < 0.5 else 1 # On choisi quelle matrice va être mise à jour
            if step == horizon-1:            
                Q[idx][state][action] += alpha*(reward - Q[idx][state][action])
            else:
                a, randomm = espilon_decreasing_greedy(action, epsilon, self.nb_action)
                target = reward + gamma * Q[1-idx][future_state][a]
                Q[idx][state][action] = (1-alpha) * Q[idx][state][action] + alpha * target
        else:
            print(f"Algo {algo} not implemented: using q_learning")
            Q, future_action = self.update_2Q(Q, step, horizon, reward, action, state, future_state, 
                       alpha, epsilon, gamma, "q_learning")
        return Q, future_action

        
    def train(self, horizon = -1, nb_episode = 100, alpha = 0.1, epsilon = 0.99, gamma = 0.99, 
              algo = "q_learning", plot = False):
        """ Fonction permettant d'entraîner notre modèle. Met à jour la matrice Q.
            
            Output: - renvoi l'évolution du nombre d'actions de chaque type par épisode
                    - plot l'évolution du reward dans le temps
        """
        ## Initialisation de Q
        if self.algo == "simple":
            Q = init_qtable(self.env, self.nb_action, self.case)
            nb_state = len(Q)
        elif self.algo == "double":
            # Retourne un tuple pour Q
            Q = (init_qtable(self.env, self.nb_action, self.case), init_qtable(self.env, self.nb_action, self.case))
            nb_state = len(Q[0])
        else:
            print("Not Implemented: algo set to 'simple'")
            self.algo = "simple"
            Q = init_qtable(self.env, self.nb_action, self.case)
            nb_state = len(Q)
            
        ## Statistiques
        actions_evolution = defaultdict(list)
        reward_evolutions = []
        
        visits = defaultdict(int) # Adaptitive alpha dictionnary
        for e in tqdm(range(nb_episode+1)):
            episode_reward = 0
            s = self.reset() # Reset de la micro grid de l'environnement et calcul de l'état
            
            ## Initialisation de l'état de départ
            if self.algo == "simple":
                a = max_dict(Q[s])[0]
            else:
                a = max_dict2(Q[0][s], Q[1][s])[0]
            a, randomm = espilon_decreasing_greedy(a, epsilon, self.nb_action)

            total_actions = {i: 0 for i in range(self.nb_action)}
            done = False
            if horizon == -1:
                horizon = 10000 # valeur arbitraire > 1 an
            for i in range(horizon):
                # Le cas où hor != horizon est le cas où on un horizon = -1 ou un horizon trop grand
                if done:
                    break
                total_actions[a] += 1
                s_, r, done, _ = self.step(a) # Mise à jour de la microgrid (récupération du reward)
                # Vérification du cas où "horizon = -1 ou horizon > max(mg.horizon)"
                if alpha == "adaptative":
                    # Mise à jour du alpha pour l'état donné
                    alpha = 1 / (1 + visits[str((s, a))])
                    visits[str((s, a))] += 1
                if self.algo == "simple":
                    Q, a_ = self.update_Q(Q, i, horizon, r, a, s, s_, alpha, epsilon, gamma, algo)
                elif self.algo == "double":
                    Q, a_ = self.update_doubleQ(Q, i, horizon, r, a, s, s_, alpha, epsilon, gamma, algo)
                s, a = s_, a_
                # Calcul des statistiques de l'épisode
                episode_reward += r
            epsilon = update_epsilon(epsilon)

            # Décompte du nombre d'actions
            for key in total_actions:
                actions_evolution[key].append(total_actions[key])
            # Moyenne du reward
            reward_evolutions.append(np.mean(episode_reward))
        self.Q = Q
        if plot:
            pd.DataFrame(reward_evolutions).plot(figsize=(12, 8), legend=False)
            plt.xlabel("Episode")
            plt.ylabel("Mean reward")
        return actions_evolution




    def test(self, horizon = -1, env=None, testing=False, verbose=False):
        if env is None:
            s = self.reset(testing=testing)
            env = self.env
        else:
            env.reset(testing=testing)
            s = self.transition(env)
        if self.algo == "simple":
            a = max_dict(self.Q[s])[0]
        else:
            a = max_dict2(self.Q[0][s], self.Q[1][s])[0]
        total_cost= 0
        if horizon == -1:
            horizon = self.max_horizon
        done = False
        if horizon == -1:
            horizon = 10000 # valeur arbitraire > 1 an
        for i in range(horizon):
            if done:
                break
            # Mise à jour de la microgrid
            if env is None:
                s_ , _ , done , _ = self.step(a)
            else:
                s_ , _ , done , _ = self.step(a, env)
                
            total_cost += env.get_cost()
            if self.algo == "simple":
                a_ = max_dict(self.Q[s_])[0]
            else:
                a_ = max_dict2(self.Q[0][s_], self.Q[1][s_])[0]            
            s, a = s_, a_
            
            if verbose:
                if i < 10:
                    print(i," -", get_name_action(a), round(env.get_cost(),1), "€")
                else:
                    print(i,"-", get_name_action(a), round(env.get_cost(),1), "€")
        return round(total_cost, 2)
    
    
def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k,v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def max_dict2(d1, d2):
    max_key = None
    max_val = float('-inf')
    for (k1,v1),(k2,v2) in zip(d1.items(),d2.items()):
        if (v1 + v2) > max_val:
            max_val = v1 + v2
            max_key = k1
    return max_key, max_val


def espilon_decreasing_greedy(action, epsilon, nb_action):
    p = np.random.random()
    if p < (1 - epsilon):
        randomm=0
        return action, randomm
    else: 
        randomm=1
        return np.random.choice(nb_action), randomm


def update_epsilon(epsilon):
    epsilon = epsilon - epsilon *0.02
    if epsilon < 0.1:
        epsilon = 0.1
    return epsilon


def init_qtable(env, nb_action, case):
    state = []
    Q = {}
    for i in range(-int(env.mg.parameters['PV_rated_power']-1),int(env.mg.parameters['load']+2)):
        for j in np.arange(round(env.mg.battery.soc_min,1),round(env.mg.battery.soc_max+0.1,1),0.1):
            j = round(j,1)
            # Ajout de la troisième dimension
            for k in range(3):
                # Dimension production supérieure à la demande
                for m in range(2):
                    # Simple building
                    if case == 0:
                        state.append((m, k,i,j)) 
                    # Building with generator
                    else:
                        for l in range(2):
                            state.append((l, m, k, i, j)) 
    #Initialize Q(s,a) at zero
    for s in state:

        Q[s] = {}

        for a in range(nb_action):

            Q[s][a] = 0
    return Q


def get_name_action(idx):
    
    #action 0: battery_charge
    #action 1: battery_discharge
    #action 2: grid_import
    #action 3: grid_export
    
    action_names = {
        0: "charge",
        1: "discharge",
        2: "import",
        3: "export",
        4: "charge from grid"
    }
    
    return action_names.get(idx, "UNKNOWN ACTION IDX")

