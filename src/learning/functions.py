import math
import os
import numpy as np
import util
from data.videoParams import VideoParams as vp

class Rep_Good:
    """
    Classe che si occupa di riconoscere e contare le ripetizioni di tutti gli esercizi.
    """

    def __init__(self):
        """
        Costruttore della classe.
        """
        self.category_data = {}
        self.parameters = {}
        self.reset()
        self.extract_parameters()


    def reset(self):
        """
        Funzione che azzera i contatori delle ripetizioni per ogni categoria e stabilisce le tolleranze per il riconoscimento dell'inizio e della fine di un esercizio.
        """

        self.category_data = {
            'arms_extension': {
                'count': 0,
                'state': 'start',
                'last_state': 'start',
                'tollerance_rep_start': 30,
                'tollerance_rep_end': 15,
                'tollerance_good_start': 10,
                'tollerance_good_end': 10,
                'start_angle': [],
                'end_angle': [],
                'current_points': -1
            },
            'arms_up': {
                'count': 0,
                'state': 'start',
                'last_state': 'start',
                'tollerance_rep_start': 20,
                'tollerance_rep_end': 20,
                'tollerance_good_start': 10,
                'tollerance_good_end': 10,
                'start_angle': [],
                'end_angle': [],
                'current_points': -1
            },
            'chair_raises': {
                'count': 0,
                'state': 'start',
                'last_state': 'start',
                'tollerance_rep_start': 35,
                'tollerance_rep_end': 15,
                'tollerance_good_start': 10,
                'tollerance_good_end': 10,
                'start_angle': [],
                'end_angle': [],
                'current_points': -1
            },
            'lateral_raises': {
                'count': 0,
                'state': 'start',
                'last_state': 'start',
                'tollerance_rep_start': 20,
                'tollerance_rep_end': 20,
                'tollerance_good_start': 10,
                'tollerance_good_end': 10,
                'start_angle': [],
                'end_angle': [],
                'current_points': -1
            },
            'leg_extension': {
                'count': 0,
                'state': 'start',
                'last_state': 'start',
                'tollerance_rep_start': 20,
                'tollerance_rep_end': 20,
                'tollerance_good_start': 10,
                'tollerance_good_end': 10,
                'start_angle': [],
                'end_angle': [],
                'current_points': -1
            }
        }

        for category in self.category_data:
            for i in range(len(vp.get_category_parameters(category)["points"])):
                self.category_data[category]["start_angle"].append(0 if vp.get_category_parameters(category)["start"] == 'min' else math.inf)
                self.category_data[category]["end_angle"].append(0 if vp.get_category_parameters(category)["start"] == 'max' else math.inf)

        self.category_phrases = {
            'arms_extension': {
                'current': "good",
                'good': "You're doing well!\nKeep it up",
                'start_over': "Don't close your arms\ntoo much",
                'start_under': "Bring your arms\ncloser together",
                'end_over': "Don't open your arms\ntoo much",
                'end_under': "Open your arms wider"
            },
            'arms_up': {
                'current': "good",
                'good': "You're doing well!\nKeep it up",
                'start_over': "You're doing well!\nKeep it up",
                'start_under': "Lower your arms more",
                'end_over': "You're doing well!\nKeep it up",
                'end_under': "Raise your arms higher"
            },
            'chair_raises': {
                'current': "good",
                'good': "You're doing well!\nKeep it up",
                'start_over': "You're doing well!\nKeep it up",
                'start_under': "Sit correctly on the chair",
                'end_over': "You're doing well!\nKeep it up",
                'end_under': "Stretch your legs\nwhen you stand up"
            },
            'lateral_raises': {
                'current': "good",
                'good': "You're doing well!\nKeep it up",
                'start_over': "Don't bring your arms\ntoo close to your body",
                'start_under': "Bring your arms closer\nto your body",
                'end_over': "Don't raise your arms\ntoo high",
                'end_under': "Raise your arms higher"
            },
            'leg_extension': {
                'current': "good",
                'good': "You're doing well!\nKeep it up",
                'start_over': "Don't close your leg\ntoo much",
                'start_under': "Close your leg more",
                'end_over': "Don't lift your leg\ntoo much",
                'end_under': "Raise your leg higher"
            }
        }


    def extract_parameters(self):
        """
        Funzione che recupera i parametri degli esercizi da file e li processa.
        """

        parameters = np.load(os.path.join(util.getParametersPath(), "parameters.npy"), allow_pickle=True).item()

        # Per ogni categoria di esercizio fa la media dei parametri di start e end
        for category in parameters:
            self.parameters[category] = []
            for i in range(len(vp.get_category_parameters(category)["points"])):
                self.parameters[category].append({
                    'start': np.mean([parameters[category][j][i]["start"] for j in range(len(parameters[category]))]),
                    'end': np.mean([parameters[category][j][i]["end"] for j in range(len(parameters[category]))])
                })


    def update(self, frame):
        """
        Funzione che aggiorna il contatore delle ripetizioni per ogni categoria di esercizio.

        Args:
        - frame (Frame): frame corrente
        """

        for category in self.category_data:
            category_type = vp.get_category_parameters(category)["type"]
            category_points = vp.get_category_parameters(category)["points"]
            category_start = vp.get_category_parameters(category)["start"]

            rep_conditions_start = []
            rep_conditions_end = []
            values = []
            for i in range(len(self.parameters[category])):
                parameters_info = {}
                if category_start == 'min':
                    interval = self.parameters[category][i]["end"] - self.parameters[category][i]["start"]
                    parameters_info = {
                        'near_start': self.parameters[category][i]["start"] + interval/100 * self.category_data[category]["tollerance_rep_start"],
                        'near_end': self.parameters[category][i]["end"] - interval/100 * self.category_data[category]["tollerance_rep_end"]
                    }
                else:
                    interval = self.parameters[category][i]["start"] - self.parameters[category][i]["end"]
                    parameters_info = {
                        'near_start': self.parameters[category][i]["start"] - interval/100 * self.category_data[category]["tollerance_rep_start"],
                        'near_end': self.parameters[category][i]["end"] + interval/100 * self.category_data[category]["tollerance_rep_end"]
                    }
                
                if category_type == 'distance':
                    value = util.calculate_distance(vp.extract_points(frame, category_points[i][0]), vp.extract_points(frame, category_points[i][1])) / util.calculate_distance(vp.extract_points(frame, 1), vp.extract_points(frame, 7))
                else:
                    value = util.calculate_angle(vp.extract_points(frame, category_points[i][0]), vp.extract_points(frame, category_points[i][1]), vp.extract_points(frame, category_points[i][2]))

                rep_conditions_start.append((self.category_data[category]["state"] == 'start') and ((category_start == 'min' and value > parameters_info['near_end']) or (category_start == 'max' and value < parameters_info['near_end'])))
                rep_conditions_end.append((self.category_data[category]["state"] == 'end') and ((category_start == 'min' and value < parameters_info['near_start']) or (category_start == 'max' and value > parameters_info['near_start'])))

                values.append(value)

            '''passed = False
            for i in range(len(rep_conditions_start)):
                if rep_conditions_start[i]:
                    self.category_data[category]["state"] = 'end'
                    self.category_data[category]["current_points"] = i
                    passed = True
                    break
            
            if not passed:
                for i in range(len(rep_conditions_end)):
                    if rep_conditions_end[i] and self.category_data[category]["current_points"] == i:
                        self.category_data[category]["state"] = 'start'
                        self.category_data[category]["count"] += 1
                        break'''

            if any(rep_conditions_start):
                self.category_data[category]["state"] = 'end'
            elif any(rep_conditions_end):
                self.category_data[category]["state"] = 'start'
                self.category_data[category]["count"] += 1

            self.update_good(category, interval, values)
                
            self.category_data[category]["last_state"] = self.category_data[category]["state"]
    

    def update_good(self, category, interval, values):
        category_start = vp.get_category_parameters(category)["start"]
        current_state = self.category_data[category]["state"]
        last_state = self.category_data[category]["last_state"]
        
        current_phrase = self.category_phrases[category]['current']

        #print(category_start, current_state, last_state)
        for angle_type_num in range(len(self.parameters[category])):
            start_angle = self.category_data[category]["start_angle"][angle_type_num]
            end_angle = self.category_data[category]["end_angle"][angle_type_num]
            start_interval = {
                'min': self.parameters[category][angle_type_num]["start"] - interval/100 * self.category_data[category]["tollerance_good_start"],
                'max': self.parameters[category][angle_type_num]["start"] + interval/100 * self.category_data[category]["tollerance_good_start"]
            }
            end_interval = {
                'min': self.parameters[category][angle_type_num]["end"] - interval/100 * self.category_data[category]["tollerance_good_end"],
                'max': self.parameters[category][angle_type_num]["end"] + interval/100 * self.category_data[category]["tollerance_good_end"]
            }

            if category_start == 'min':
                if current_state == 'start':
                    # Se si è appena entrati nello stato start, si deve controllare l'angolo di end
                    if last_state == 'end':
                        # Se l'angolo maggiore rilevato è minore dell'intervallo di tolleranza -> end_under (se non ci sono altri problemi), se è maggiore -> end_over (se non ci sono altri problemi), altrimenti è good (se il problema è stato risolto)
                        if end_angle < end_interval['min']:
                            self.category_phrases[category]['current'] = 'end_under' if current_phrase == 'good' or current_phrase == 'end_over' else current_phrase
                        elif end_angle > end_interval['max']:
                            self.category_phrases[category]['current'] = 'end_over' if current_phrase == 'good' or current_phrase == 'end_under' else current_phrase
                        else:
                            self.category_phrases[category]['current'] = 'good' if current_phrase == 'end_over' or current_phrase == 'end_under' else current_phrase
                        # Si azzera l'angolo di end
                        self.category_data[category]["end_angle"][angle_type_num] = 0
                    if values[angle_type_num] < self.category_data[category]["start_angle"][angle_type_num]:
                        self.category_data[category]["start_angle"][angle_type_num] = values[angle_type_num]
                elif current_state == 'end':
                    # Se si è appena entrati nello stato end, si deve controllare l'angolo di start
                    if last_state == 'start':
                        # Se l'angolo minore rilevato è maggiore dell'intervallo di tolleranza -> start_under (se non ci sono altri problemi), se è minore -> start_over (se non ci sono altri problemi), altrimenti è good (se il problema è stato risolto)
                        if start_angle > start_interval['max']:
                            self.category_phrases[category]['current'] = 'start_under' if current_phrase == 'good' or current_phrase == 'start_over' else current_phrase
                        elif start_angle < start_interval['min']:
                            self.category_phrases[category]['current'] = 'start_over' if current_phrase == 'good' or current_phrase == 'start_under' else current_phrase
                        else:
                            self.category_phrases[category]['current'] = 'good' if current_phrase == 'start_over' or current_phrase == 'start_under' else current_phrase
                        # Si azzera l'angolo di start
                        self.category_data[category]["start_angle"][angle_type_num] = math.inf
                    if values[angle_type_num] > self.category_data[category]["end_angle"][angle_type_num]:
                        self.category_data[category]["end_angle"][angle_type_num] = values[angle_type_num]
            else:
                if current_state == 'start':
                    # Se si è appena entrati nello stato start, si deve controllare l'angolo di end
                    if last_state == 'end':
                        # Se l'angolo minore rilevato è maggiore dell'intervallo di tolleranza -> end_under (se non ci sono altri problemi), se è minore -> end_over (se non ci sono altri problemi), altrimenti è good (se il problema è stato risolto)
                        if end_angle > end_interval['max']:
                            self.category_phrases[category]['current'] = 'end_under' if current_phrase == 'good' or current_phrase == 'end_over' else current_phrase
                        elif end_angle < end_interval['min']:
                            self.category_phrases[category]['current'] = 'end_over' if current_phrase == 'good' or current_phrase == 'end_under' else current_phrase
                        else:
                            self.category_phrases[category]['current'] = 'good' if current_phrase == 'end_over' or current_phrase == 'end_under' else current_phrase
                        # Si azzera l'angolo di end
                        self.category_data[category]["end_angle"][angle_type_num] = math.inf
                    if values[angle_type_num] > self.category_data[category]["start_angle"][angle_type_num]:
                        self.category_data[category]["start_angle"][angle_type_num] = values[angle_type_num]
                elif current_state == 'end':
                    # Se si è appena entrati nello stato end, si deve controllare l'angolo di start
                    if last_state == 'start':
                        # Se l'angolo maggiore rilevato è minore dell'intervallo di tolleranza -> start_under (se non ci sono altri problemi), se è maggiore -> start_over (se non ci sono altri problemi), altrimenti è good (se il problema è stato risolto)
                        if start_angle < start_interval['min']:
                            self.category_phrases[category]['current'] = 'start_under' if current_phrase == 'good' or current_phrase == 'start_over' else current_phrase
                        elif start_angle > start_interval['max']:
                            self.category_phrases[category]['current'] = 'start_over' if current_phrase == 'good' or current_phrase == 'start_under' else current_phrase
                        else:
                            self.category_phrases[category]['current'] = 'good' if current_phrase == 'start_over' or current_phrase == 'start_under' else current_phrase
                        # Si azzera l'angolo di start
                        self.category_data[category]["start_angle"][angle_type_num] = 0
                    if values[angle_type_num] < self.category_data[category]["end_angle"][angle_type_num]:
                        self.category_data[category]["end_angle"][angle_type_num] = values[angle_type_num]

    
    # FUNZIONI GET E SET

    def get_category_rep(self, category):
        """
        Funzione che restituisce il contatore delle ripetizioni di una categoria.

        Args:
        - category (String): categoria dell'esercizio

        Returns:
        - count (Integer): numero di ripetizioni
        """

        return self.category_data[category]["count"]
    
    def get_category_phrase(self, category):
        """
        Funzione che restituisce la frase corrente per una categoria.

        Args:
        - category (String): categoria dell'esercizio

        Returns:
        - phrase (String): frase corrente
        """

        return self.category_phrases[category][self.category_phrases[category]['current']]
    
    def reset_category_count(self, category):
        """
        Funzione che azzera il contatore delle ripetizioni di una categoria.

        Args:
        - category (String): categoria dell'esercizio
        """

        self.category_data[category]["count"] = 0
        self.category_data[category]["state"] = 'start'
        self.category_data[category]["last_state"] = 'start'
        self.category_data[category]["current_points"] = -1