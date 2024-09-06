import os
import numpy as np
import util
from data.videoParams import VideoParams as vp

class Repetitions:
    """
    Classe che si occupa di riconoscere e contare le ripetizioni di tutti gli esercizi.
    """

    def __init__(self):
        """
        Costruttore della classe.
        """
        self.category_rep = {}
        self.parameters = {}
        self.reset()
        self.extract_parameters()


    def reset(self):
        """
        Funzione che azzera i contatori delle ripetizioni per ogni categoria e stabilisce le tolleranze per il riconoscimento dell'inizio e della fine di un esercizio.
        """

        self.category_rep = {
            'arms_extension': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 30,
                'tollerance_end': 15
            },
            'arms_up': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 20,
                'tollerance_end': 20
            },
            'chair_raises': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 35,
                'tollerance_end': 15
            },
            'lateral_raises': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 20,
                'tollerance_end': 20
            },
            'leg_extension': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 20,
                'tollerance_end': 20
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

        for category in self.category_rep:
            category_type = vp.get_category_parameters(category)["type"]
            category_points = vp.get_category_parameters(category)["points"]
            category_start = vp.get_category_parameters(category)["start"]

            rep_conditions_start = []
            rep_conditions_end = []
            for i in range(len(self.parameters[category])):
                parameters_info = {}
                if category_start == 'min':
                    interval = self.parameters[category][i]["end"] - self.parameters[category][i]["start"]
                    parameters_info = {
                        'near_start': self.parameters[category][i]["start"] + interval/100 * self.category_rep[category]["tollerance_start"],
                        'near_end': self.parameters[category][i]["end"] - interval/100 * self.category_rep[category]["tollerance_end"]
                    }
                else:
                    interval = self.parameters[category][i]["start"] - self.parameters[category][i]["end"]
                    parameters_info = {
                        'near_start': self.parameters[category][i]["start"] - interval/100 * self.category_rep[category]["tollerance_start"],
                        'near_end': self.parameters[category][i]["end"] + interval/100 * self.category_rep[category]["tollerance_end"]
                    }
                
                if category_type == 'distance':
                    value = util.calculate_distance(vp.extract_points(frame, category_points[i][0]), vp.extract_points(frame, category_points[i][1])) / util.calculate_distance(vp.extract_points(frame, 1), vp.extract_points(frame, 7))
                else:
                    value = util.calculate_angle(vp.extract_points(frame, category_points[i][0]), vp.extract_points(frame, category_points[i][1]), vp.extract_points(frame, category_points[i][2]))

                rep_conditions_start.append((self.category_rep[category]["state"] == 'start') and ((category_start == 'min' and value > parameters_info['near_end']) or (category_start == 'max' and value < parameters_info['near_end'])))
                rep_conditions_end.append((self.category_rep[category]["state"] == 'end') and ((category_start == 'min' and value < parameters_info['near_start']) or (category_start == 'max' and value > parameters_info['near_start'])))

            if any(rep_conditions_start):
                self.category_rep[category]["state"] = 'end'
            elif any(rep_conditions_end):
                self.category_rep[category]["state"] = 'start'
                self.category_rep[category]["count"] += 1

    
    # FUNZIONI GET E SET

    def get_category_rep(self, category):
        """
        Funzione che restituisce il contatore delle ripetizioni di una categoria.

        Args:
        - category (String): categoria dell'esercizio

        Returns:
        - count (Integer): numero di ripetizioni
        """

        return self.category_rep[category]["count"]