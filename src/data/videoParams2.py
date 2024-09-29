import util
import numpy as np

class VideoParams:
    """
    Classe che estrae e processa i parametri di un esercizio.
    """

    parameters = {  # parametri degli esercizi: tipo di dato da estrarre, distanza (minima o massima di partenza), punti da cui estrarre i dati
        'arms_extension': {
            'type': 'angle',
            'start': 'min',
            'points': [
                [3, (7, 8), 4]
            ]
        },
        'arms_up': {
            'type': 'angle',
            'start': 'max',
            'points': [
                [5, (9, 10), 6]
            ]
        },
        'chair_raises': {
            'type': 'angle',
            'start': 'min',
            'points': [
                [8, 7, 10]
            ]
        },
        'lateral_raises': {
            'type': 'angle',
            'start': 'min',
            'points': [
                [3, 1, 7],
                [4, 2, 8]
            ]
        },
        'leg_extension': {
            'type': 'angle',
            'start': 'min',
            'points': [
                [7, 8, 10],
                [8, 7, 9]
            ]
        }
    }


    def __init__(self, frames, category):
        """
        Costruttore della classe.
        Pende in input i frame che compongono il video e la categoria dell'esercizio.

        Args:
        - frames (Array): array di frame
        - category (String): categoria dell'esercizio
        """

        self.frames = frames
        self.category = category
        self.params = []

    
    def extract_parameters(self):
        """
        Funzione che estrae i parametri dell'esercizio.
        """

        type = VideoParams.parameters[self.category]['type']
        if type == 'distance':
            self.extract_distance()
        elif type == 'angle':
            self.extract_angle()
    

    def extract_distance(self):
        """
        Funzione che estrae la distanza tra due punti in tutti i frame.
        """

        points = VideoParams.parameters[self.category]['points']
        for point in points:
            param = []
            for frame in self.frames:
                param.append(util.calculate_distance(self.extract_points(frame, point[0]), self.extract_points(frame, point[1])) / util.calculate_distance(self.extract_points(frame, 1), self.extract_points(frame, 7)))
            self.params.append(param)
        
    
    def extract_angle(self):
        """
        Funzione che estrae l'angolo in tutti i frame.
        """

        points = VideoParams.parameters[self.category]['points']
        for point in points:
            param = []
            for frame in self.frames:
                param.append(util.calculate_angle(self.extract_points(frame, point[0]), self.extract_points(frame, point[1]), self.extract_points(frame, point[2])))
            self.params.append(param)


    def process_parameters(self, num_samples=2):
        """
        Funzione che trova i valori di inizio e fine esercizio.
        Per trovarli viene eseguita la media dei picchi dei parametri.

        Returns:
        - processed_params (Array): parametri processati
        - num_samples (int): numero di campioni da considerare
        """

        start = VideoParams.parameters[self.category]['start']
        processed_params = []

        for i in range(len(self.params)):  # Per ogni dato (angoli o distanze) dell'esercizio
            max_values = sorted(self.params[i], reverse=True)[:num_samples]
            min_values = sorted(self.params[i])[:num_samples]
            if start == 'min':
                processed_params.append({
                    'start': np.mean(min_values),
                    'end': np.mean(max_values)
                })
            else:
                processed_params.append({
                    'start': np.mean(max_values),
                    'end': np.mean(min_values)
                })

        return processed_params
    

    @staticmethod
    def extract_points(frame, p):
        """
        Funzione statica che restituisce il punto p-esimo del frame.
        Se il punto è una tupla, restituisce il punto medio tra i due punti.

        Args:
        - frame (Frame): frame da cui estrarre il punto
        - p (int, Tuple): punto

        Returns:
        - point (dict): punto
        """
        
        if type(p) == int:
            return frame.get_keypoint(p)
        else:
            a = p[0]
            b = p[1]
            return {
                'x': (frame.get_keypoint(a)['x'] + frame.get_keypoint(b)['x']) / 2,
                'y': (frame.get_keypoint(a)['y'] + frame.get_keypoint(b)['y']) / 2
            }
    

    # FUNZIONI GET E SET

    @staticmethod
    def get_category_parameters(category):
        """
        Funzione che restituisce la forma dei parametri di una categoria.

        Args:
        - category (String): categoria dell'esercizio

        Returns:
        - parameters (dict): parametri dell'esercizio
        """

        return VideoParams.parameters[category]