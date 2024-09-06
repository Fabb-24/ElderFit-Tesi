import os
import numpy as np
import util
from data.videoParams import VideoParams as vp

class Repetitions:

    def __init__(self):
        self.category_rep = {}
        self.parameters = {}
        self.reset()
        self.extract_parameters()


    def reset(self):
        """
        Funzione che azzera i contatori delle ripetizioni per ogni categoria.
        """

        self.category_rep = {
            'arms_extension': {
                'count': 0,
                'state': 'start',
                'tollerance_start': 20,
                'tollerance_end': 20
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
                'tollerance_start': 30,
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
            params_type = vp.get_category_parameters(category)["type"]
            params_points = vp.get_category_parameters(category)["points"]
            params_start = vp.get_category_parameters(category)["start"]

            if params_start == 'min':
                #near_interval = ((self.parameters[category][0]["end"] - self.parameters[category][0]["start"]) / 100) * 20
                interval = self.parameters[category][0]["end"] - self.parameters[category][0]["start"]
                near_start = self.parameters[category][0]["start"] + interval/100 * self.category_rep[category]["tollerance_start"]
                near_end = self.parameters[category][0]["end"] - interval/100 * self.category_rep[category]["tollerance_end"]
            elif params_start == 'max':
                #near_interval = ((self.parameters[category][0]["start"] - self.parameters[category][0]["end"]) / 100) * 20
                interval = self.parameters[category][0]["start"] - self.parameters[category][0]["end"]
                near_start = self.parameters[category][0]["start"] - interval/100 * self.category_rep[category]["tollerance_start"]
                near_end = self.parameters[category][0]["end"] + interval/100 * self.category_rep[category]["tollerance_end"]

            if params_type == 'distance':
                #value = util.calculate_distance(frame.get_keypoint(params_points[0][0]), frame.get_keypoint(params_points[0][1])) / util.calculate_distance(frame.get_keypoint(1), frame.get_keypoint(7))
                value = util.calculate_distance(vp.extract_points(frame, params_points[0][0]), vp.extract_points(frame, params_points[0][1])) / util.calculate_distance(vp.extract_points(frame, 1), vp.extract_points(frame, 7))
            elif params_type == 'angle':
                #value = util.calculate_angle(frame.get_keypoint(params_points[0][0]), frame.get_keypoint(params_points[0][1]), frame.get_keypoint(params_points[0][2]))
                value = util.calculate_angle(vp.extract_points(frame, params_points[0][0]), vp.extract_points(frame, params_points[0][1]), vp.extract_points(frame, params_points[0][2]))

            '''if category == "chair_raises":
                print(f"Value: {value}, Near start: {near_start}, Near end: {near_end}")'''

            if self.category_rep[category]["state"] == 'start':
                if (params_start == 'min' and value > near_end) or (params_start == 'max' and value < near_end):
                    self.category_rep[category]["state"] = 'end'
            if self.category_rep[category]["state"] == 'end':
                if (params_start == 'min' and value < near_start) or (params_start == 'max' and value > near_start):
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