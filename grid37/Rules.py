class Naive_algo():
    
    """
    Class to use the 3 naives based rules algorithms
    If building 3 -> go to algo_2
    Else -> go to either algo_0 or algo_1 (algo_1 by default)
    """
    
    def __init__(self, num=1):
        self.algo = num
    
    def compute(self, building):
        
        building.reset()
        
        if building.architecture['genset'] == 1:
            return algo_2(building)
        else:
            if self.algo == 0:
                return algo_0(building)
            elif self.algo != 0:
                return algo_1(building)
        
        building.reset()
        

def algo_0(building):
    
    """
    Most naive approach
    """
    
    building_data = building.get_updated_values()
    total_building_cost = 0
    
    while building.done == False:        
        
        load = building_data['load']
        pv = building_data['pv']
        p_char = max(0, min(pv-load, building_data['capa_to_charge'], building.battery.p_charge_max))
        p_dischar = max(0, min(load-pv, building_data['capa_to_discharge'], building.battery.p_discharge_max))

        control_dict = {'battery_charge': p_char,
                         'battery_discharge': p_dischar,
                         'grid_import': max(0, load-pv)-p_dischar,
                         'grid_export': max(0, pv-load)-p_char,
                         'pv_consummed': pv,
                         'genset': 0
                         }
        
        building_data = building.run(control_dict)
        total_building_cost += building.get_cost()
        
    return total_building_cost


def algo_1(building):
    
    """
    Strategy : try to fill the battery when the import cost is low to prevent doing so when prices are higher in
               a near future
    """
    
    # initialize values
    building_data = building.get_updated_values() 
    total_building_cost = 0 
    capa_max = building_data['capa_to_charge']
    
    # loop on hours
    while building.done == False:   
        
        load = building_data['load']
        pv = building_data['pv']
        capa_to_charge = building_data['capa_to_charge']
        capa_to_discharge = building_data['capa_to_discharge']
        hour = building_data['hour']
        
        p_char = max(0, min(pv-load, capa_to_charge, building.battery.p_charge_max))
        p_dischar = max(0, min(load-pv, capa_to_discharge, building.battery.p_discharge_max))

        control_dict = {'battery_charge': p_char,
                        'battery_discharge': p_dischar,
                        'grid_import': max(0, load-pv-p_dischar),
                        'grid_export': max(0, pv-load-p_char),
                        'pv_consummed': pv,
                        'genset': 0
                        }
        
        forecast_load = building.forecast_load()
        forecast_pv = building.forecast_pv()
        
        if hour==0:
            need_next_day_expensive = (forecast_load - forecast_pv)[11:17].sum()
            need_next_day_intermediary_left = (forecast_load - forecast_pv)[7:11].sum()
        
        if need_next_day_expensive > capa_max:
            if hour in [4, 5, 6, 7]:
                p_char = min(need_next_day_expensive, capa_to_charge, building.battery.p_charge_max) 
                control_dict = {'battery_charge': p_char,
                                'battery_discharge': 0,
                                'grid_import': max(0, load+p_char-pv),
                                'grid_export': max(0, pv-load-p_char),
                                'pv_consummed': pv,
                                'genset': 0
                         }
            if hour in [8, 9, 10, 11]:
                control_dict = {'battery_charge': 0,
                                'battery_discharge': 0,
                                'grid_import': max(0, load-pv),
                                'grid_export': max(0, pv-load),
                                'pv_consummed': pv,
                                'genset': 0
                         }
                
        if need_next_day_expensive <= capa_max and need_next_day_expensive > 0:
            
            if need_next_day_intermediary_left > 0:
            
                if hour in [4, 5, 6, 7]:
                    p_char = min((need_next_day_expensive + need_next_day_intermediary_left)/4, capa_to_charge, 
                                 building.battery.p_charge_max) 
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': max(0, load+p_char-pv),
                                    'grid_export': max(0, pv-load-p_char),
                                    'pv_consummed': pv,
                                    'genset': 0
                             }
                    
            else:
                
                if hour in [4, 5, 6, 7]:
                    p_char = max(0, min((need_next_day_expensive + need_next_day_intermediary_left)/4, 
                                        capa_to_charge, building.battery.p_charge_max)) 
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': max(0, load+p_char-pv),
                                    'grid_export': max(0, pv-load-p_char),
                                    'pv_consummed': pv,
                                    'genset': 0
                             }

        building_data = building.run(control_dict)
        total_building_cost += building.get_cost()
        
    return total_building_cost


def algo_2(building):
    
    """
    Especialy for the 3rd building
    """

    building_data = building.get_updated_values()
    total_building_cost = 0

    while building.done == False:
        
        grid_status = building_data['grid_status']
        load = building_data['load']
        pv = building_data['pv']
        capa_to_charge = building_data['capa_to_charge']
        capa_to_discharge = building_data['capa_to_discharge']
        hour = building_data['hour']

        if pv > load: # battery or curtail
            p_char = max(0, min(pv - load, capa_to_charge, building.battery.p_charge_max))
            control_dict = {'battery_charge': p_char,
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': 0, 
                            'pv_consummed': load+p_char,
                            'pv_curtailed': max(0, pv-p_char-load),
                            'genset': 0
                           } 

        if pv <= load: # battery or import or genset
            
            if grid_status == 1:
                
                if building.forecast_grid_status()[4] == 1: #if in 4 hours we are still in state 1, we can use battery
                    p_disc = max(0, min(load-pv, capa_to_discharge, building.battery.p_discharge_max))
                    control_dict = {'battery_charge': 0,
                                    'battery_discharge': p_disc,
                                    'grid_import': max(0, load-pv-p_disc),
                                    'grid_export': 0,
                                    'pv_consummed': min(pv, load),
                                    'genset': 0
                                   } 
        
                else: # charge battery if 0s are coming in 4 hours so it will be full
                    p_char = max(0, min(capa_to_charge, building.battery.p_charge_max))
                    control_dict = {'battery_charge': p_char,
                                    'battery_discharge': 0,
                                    'grid_import': max(0, load-pv+p_char),
                                    'grid_export': 0,
                                    'pv_consummed': min(pv, load),
                                    'genset': 0
                                   }
            
            else:                    
                p_disc = max(0, min(load-pv, capa_to_discharge, building.battery.p_discharge_max))
                control_dict = {'battery_charge': 0,
                                'battery_discharge': p_disc,
                                'grid_import': 0,
                                'grid_export': 0,
                                'pv_consummed': min(pv, load),
                                'genset': max(0, load-pv-p_disc)
                               }
                pass

        building_data = building.run(control_dict)
        total_building_cost += building.get_cost()

    return total_building_cost