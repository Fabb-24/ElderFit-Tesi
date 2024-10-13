import math
import os
import numpy as np
import util

from data.videoParams import VideoParams as vp

class Functions:
    """
    Classe che si occupa del conteggio delle ripetizioni e della generazione di feedback per l'utente.
    """

    def __init__(self):
        """
        Costruttore della classe. Inizializzo le variabili necessarie.
        """

        self.repetitions = {}
        self.reset_repetitions()
        self.parameters = {}
        self.extract_parameters()

        self.executions = {
            'arms_extension': {
                'max_angle': [],
                'min_angle': [],
                'reverse': False
            },
            'arms_up': {
                'max_angle': [],
                'min_angle': [],
                'reverse': True
            },
            'chair_raises': {
                'max_angle': [],
                'min_angle': [],
                'reverse': False
            },
            'lateral_raises': {
                'max_angle': [],
                'min_angle': [],
                'reverse': False
            },
            'leg_extension': {
                'max_angle': [],
                'min_angle': [],
                'reverse': False
            }
        }

        for category in self.executions.keys():
            self.executions[category]['max_angle'] = [0 for i in range(len(vp.category_angles[category]))]
            self.executions[category]['min_angle'] = [180 for i in range(len(vp.category_angles[category]))]

        self.feedbacks = {
        'arms_extension': {
            'current': "good",
            'good': ("You're doing well!\nKeep it up", True),
            'start_over': ("Don't close your arms\ntoo much", False),
            'start_under': ("Bring your arms\ncloser together", False),
            'end_over': ("Don't open your arms\ntoo much", False),
            'end_under': ("Open your arms wider", False)
        },
        'arms_up': {
            'current': "good",
            'good': ("You're doing well!\nKeep it up", True),
            'start_over': ("You're doing well!\nKeep it up", True),
            'start_under': ("Lower your arms more", False),
            'end_over': ("You're doing well!\nKeep it up", True),
            'end_under': ("Raise your arms higher", False)
        },
        'chair_raises': {
            'current': "good",
            'good': ("You're doing well!\nKeep it up", True),
            'start_over': ("You're doing well!\nKeep it up", True),
            'start_under': ("Sit correctly on the chair", False),
            'end_over': ("You're doing well!\nKeep it up", True),
            'end_under': ("Stretch your legs\nwhen you stand up", False)
        },
        'lateral_raises': {
            'current': "good",
            'good': ("You're doing well!\nKeep it up", True),
            'start_over': ("Don't bring your arms\ntoo close to your body", False),
            'start_under': ("Bring your arms closer\nto your body", False),
            'end_over': ("Don't raise your arms\ntoo high", False),
            'end_under': ("Raise your arms higher", False)
        },
        'leg_extension': {
            'current': "good",
            'good': ("You're doing well!\nKeep it up", True),
            'start_over': ("Don't close your leg\ntoo much", False),
            'start_under': ("Close your leg more", False),
            'end_over': ("Don't lift your leg\ntoo much", False),
            'end_under': ("Raise your leg higher", False)
        }
    }


    def reset_repetitions(self):
        """
        Resetto il conteggio delle ripetizioni per tutte le categorie di esercizi.
        """

        self.repetitions = {
            'arms_extension': {
                'count': 0,
                'state': 'start',
                'start_time': 0,
                'times': [],
                'accuracy': (0, 0)
            },
            'arms_up': {
                'count': 0,
                'state': 'start',
                'start_time': 0,
                'times': [],
                'accuracy': (0, 0)
            },
            'chair_raises': {
                'count': 0,
                'state': 'start',
                'start_time': 0,
                'times': [],
                'accuracy': (0, 0)
            },
            'lateral_raises': {
                'count': 0,
                'state': 'start',
                'start_time': 0,
                'times': [],
                'accuracy': (0, 0)
            },
            'leg_extension': {
                'count': 0,
                'state': 'start',
                'start_time': 0,
                'times': [],
                'accuracy': (0, 0)
            }
        }


    def extract_parameters(self):
        self.parameters = np.load(os.path.join(util.getParametersPath(), "parameters.npy"), allow_pickle=True).item()


    def update(self, frame):
        """
        Funzione che aggiorna il conteggio delle ripetizioni e la generazione di feedback.

        Args:
        - frame (numpy.ndarray): frame da processare
        """

        self.update_repetitions(frame)
        self.update_feedbacks(frame)

    
    def update_repetitions(self, frame):
        """
        Aggiorno il numero di ripetizioni per ogni categoria di esercizio.

        Args:
            frame (Frame): frame da processare
        """

        curr_keypoints = frame.process_keypoints().tolist()

        for category in self.repetitions.keys():
            distance_max = min([self.keypoints_distance(curr_keypoints, keypoints) for keypoints in self.parameters[category]["keypoints_max"]])
            distance_min = min([self.keypoints_distance(curr_keypoints, keypoints) for keypoints in self.parameters[category]["keypoints_min"]])

            if not self.executions[category]['reverse']:
                if self.repetitions[category]['state'] == 'start':
                    if distance_max < distance_min:
                        self.repetitions[category]['state'] = 'end'
                        self.executions[category]['max_angle'] = [0 for i in range(len(vp.category_angles[category]))]
                        self.update_feedback_msg(category)
                elif self.repetitions[category]['state'] == 'end':
                    if distance_min < distance_max:
                        self.repetitions[category]['count'] += 1
                        self.repetitions[category]['state'] = 'start'
                        self.executions[category]['min_angle'] = [180 for i in range(len(vp.category_angles[category]))]
                        self.update_feedback_msg(category)
                        self.update_times(category)
            else:
                if self.repetitions[category]['state'] == 'start':
                    if distance_max > distance_min:
                        self.repetitions[category]['state'] = 'end'
                        self.executions[category]['min_angle'] = [180 for i in range(len(vp.category_angles[category]))]
                        self.update_feedback_msg(category)
                elif self.repetitions[category]['state'] == 'end':
                    if distance_min > distance_max:
                        self.repetitions[category]['count'] += 1
                        self.repetitions[category]['state'] = 'start'
                        self.executions[category]['max_angle'] = [0 for i in range(len(vp.category_angles[category]))]
                        self.update_feedback_msg(category)
                        self.update_times(category)

    
    def update_times(self, category):
        """
        Aggiorno i tempi di esecuzione per una categoria specifica. Registro il tempo di esecuzione della ripetizione e resetto il tempo di inizio.

        Args:
        - category (String): categoria dell'esercizio
        """

        if self.repetitions[category]['count'] == 1:
            self.repetitions[category]['start_time'] = util.get_current_time()
        elif self.repetitions[category]['count'] > 1:
            self.repetitions[category]['times'].append(util.get_current_time() - self.repetitions[category]['start_time'])
            self.repetitions[category]['start_time'] = util.get_current_time()


    def update_feedbacks(self, frame):
        """
        Funzione che aggiorna i feedback per l'utente.

        Args:
        - frame (numpy.ndarray): frame da processare
        """

        for category in self.repetitions.keys():
            angles_points = vp.category_angles[category]

            for angle_index in range(len(angles_points)):
                curr_angle = util.calculate_angle(vp.extract_points(frame, angles_points[angle_index][0]), vp.extract_points(frame, angles_points[angle_index][1]), vp.extract_points(frame, angles_points[angle_index][2]))

                if self.repetitions[category]['state'] == 'start':
                    if not self.executions[category]['reverse'] and curr_angle < self.executions[category]["min_angle"][angle_index]:
                        self.executions[category]['min_angle'][angle_index] = curr_angle
                    elif self.executions[category]['reverse'] and curr_angle > self.executions[category]["max_angle"][angle_index]:
                        self.executions[category]['max_angle'][angle_index] = curr_angle
                elif self.repetitions[category]['state'] == 'end':
                    if not self.executions[category]['reverse'] and curr_angle > self.executions[category]["max_angle"][angle_index]:
                        self.executions[category]['max_angle'][angle_index] = curr_angle
                    elif self.executions[category]['reverse'] and curr_angle < self.executions[category]["min_angle"][angle_index]:
                        self.executions[category]['min_angle'][angle_index] = curr_angle


    def update_feedback_msg(self, category, tollerance=5):
        tollerance = tollerance / 100

        angles_points = vp.category_angles[category]

        for angle_index in range(len(angles_points)):
            interval = (self.parameters[category]["angles_max"][angle_index] - self.parameters[category]["angles_min"][angle_index]) * tollerance
            self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1)

            if self.repetitions[category]['state'] == 'start':
                if not self.executions[category]['reverse']:
                    if self.executions[category]['max_angle'][angle_index] < self.parameters[category]["angles_max"][angle_index] - interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'end_under' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'end_over' else self.feedbacks[category]['current']
                    elif self.executions[category]['max_angle'][angle_index] > self.parameters[category]["angles_max"][angle_index] + interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'end_over' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'end_under' else self.feedbacks[category]['current']
                    else:
                        self.feedbacks[category]['current'] = 'good' if self.feedbacks[category]['current'] == 'end_over' or self.feedbacks[category]['current'] == 'end_under' else self.feedbacks[category]['current']
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0] + 1, self.repetitions[category]['accuracy'][1], self.repetitions[category]['accuracy'][2])
                else:
                    if self.executions[category]['min_angle'][angle_index] > self.parameters[category]["angles_min"][angle_index] + interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'end_under' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'end_over' else self.feedbacks[category]['current']
                    elif self.executions[category]['min_angle'][angle_index] < self.parameters[category]["angles_min"][angle_index] - interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'end_over' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'end_under' else self.feedbacks[category]['current']
                    else:
                        self.feedbacks[category]['current'] = 'good' if self.feedbacks[category]['current'] == 'end_over' or self.feedbacks[category]['current'] == 'end_under' else self.feedbacks[category]['current']
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0] + 1, self.repetitions[category]['accuracy'][1], self.repetitions[category]['accuracy'][2])
            elif self.repetitions[category]['state'] == 'end':
                if not self.executions[category]['reverse']:
                    if self.executions[category]['min_angle'][angle_index] > self.parameters[category]["angles_min"][angle_index] + interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'start_under' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'start_over' else self.feedbacks[category]['current']
                    elif self.executions[category]['min_angle'][angle_index] < self.parameters[category]["angles_min"][angle_index] - interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'start_over' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'start_under' else self.feedbacks[category]['current']
                    else:
                        self.feedbacks[category]['current'] = 'good' if self.feedbacks[category]['current'] == 'start_over' or self.feedbacks[category]['current'] == 'start_under' else self.feedbacks[category]['current']
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0] + 1, self.repetitions[category]['accuracy'][1], self.repetitions[category]['accuracy'][2])
                else:
                    if self.executions[category]['max_angle'][angle_index] < self.parameters[category]["angles_max"][angle_index] - interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'start_under' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'start_over' else self.feedbacks[category]['current']
                    elif self.executions[category]['max_angle'][angle_index] > self.parameters[category]["angles_max"][angle_index] + interval:
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0], self.repetitions[category]['accuracy'][1] + 1, self.repetitions[category]['accuracy'][2])
                        self.feedbacks[category]['current'] = 'start_over' if self.feedbacks[category]['current'] == 'good' or self.feedbacks[category]['current'] == 'start_under' else self.feedbacks[category]['current']
                    else:
                        self.feedbacks[category]['current'] = 'good' if self.feedbacks[category]['current'] == 'start_over' or self.feedbacks[category]['current'] == 'start_under' else self.feedbacks[category]['current']
                        #self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0] + 1, self.repetitions[category]['accuracy'][1], self.repetitions[category]['accuracy'][2])
            
            if self.feedbacks[category][self.feedbacks[category]['current']][1]:
                self.repetitions[category]['accuracy'] = (self.repetitions[category]['accuracy'][0] + 1, self.repetitions[category]['accuracy'][1])

            if category == 'arms_up':
                print(self.repetitions[category]['accuracy'])



    def keypoints_distance(self, kp1, kp2):
        """
        Calcolo la distanza tra due insiemi di keypoints

        Args:
            kp1 (list): primo insieme di keypoints (lista semplice di valori)
            kp2 (list): secondo insieme di keypoints (lista semplice di valori)

        Returns:
            float: distanza tra i due insiemi di keypoints
        """

        sum = 0
        for i in range(len(kp1)):
            sum += (kp1[i] - kp2[i]) ** 2
        return math.sqrt(sum)


    def reset_category_repetitions(self, category):
        """
        Resetto il conteggio delle ripetizioni per una categoria specifica.

        Args:
            category (String): categoria dell'esercizio
        """

        self.repetitions[category]['count'] = 0
        self.repetitions[category]['state'] = 'start'
        self.repetitions[category]['start_time'] = 0
        self.repetitions[category]['times'] = []
        self.repetitions[category]['accuracy'] = (0, 0)


    def get_category_repetitions(self, category):
        """
        Restituisco il numero di ripetizioni per una categoria specifica.

        Args:
            category (String): categoria dell'esercizio

        Returns:
            int: numero di ripetizioni
        """

        return self.repetitions[category]['count']
    

    def get_category_avg_time(self, category):
        """
        Restituisco il tempo medio di esecuzione per una categoria specifica.

        Args:
            category (String): categoria dell'esercizio

        Returns:
            float: tempo medio di esecuzione
        """

        return np.mean(self.repetitions[category]['times']) if len(self.repetitions[category]['times']) > 0 else 0


    def get_category_accuracy(self, category):
        """
        Restituisco l'accuratezza dell'esecuzione per una categoria specifica.

        Args:
            category (String): categoria dell'esercizio

        Returns:
            float: accuratezza dell'esecuzione
        """

        return self.repetitions[category]['accuracy'][0] / self.repetitions[category]['accuracy'][1] if self.repetitions[category]['accuracy'][1] > 0 else 0
    

    def get_category_phrase(self, category):
        """
        Restituisco il feedback associato all'esercizio.

        Args:
            category (String): categoria dell'esercizio

        Returns:
            String: feedback associato all'esercizio
        """

        return self.feedbacks[category][self.feedbacks[category]['current']][0]