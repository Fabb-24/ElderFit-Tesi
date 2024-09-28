import util

class VideoParams:

    category_angles = {
        'arms_extension': [
            [3, (7, 8), 4]
        ],
        'arms_up': [
            [5, (9, 10), 6]
        ],
        'chair_raises': [
            [8, 7, 10]
        ],
        'lateral_raises': [
            [3, 1, 7],
            [4, 2, 8]
        ],
        'leg_extension': [
            [7, 8, 10],
            [8, 7, 9]
        ]
    }


    def __init__(self, frames, category):
        self.frames = frames
        self.category = category

    
    def extract_parameters(self):
        frames_max = []
        frames_min = []

        for angle_points in self.category_angles[self.category]:
            angles = []

            for i in range(len(self.frames)):
                angle = util.calculate_angle(self.extract_points(self.frames[i], angle_points[0]), self.extract_points(self.frames[i], angle_points[1]), self.extract_points(self.frames[i], angle_points[2]))
                angles.append(angle)

            # Trovo l'indice di angles in cui c'è l'angolo più ampio e quello più stretto
            max_angle_index = angles.index(max(angles))
            min_angle_index = angles.index(min(angles))

            # Aggiungo i frame corrispondenti agli indici trovati
            frames_max.append(self.frames[max_angle_index])
            frames_min.append(self.frames[min_angle_index])

        keypoints_max = [frame.process_keypoints() for frame in frames_max]
        keypoints_min = [frame.process_keypoints() for frame in frames_min]
        angles_max = [frame.process_angles() for frame in frames_max]
        angles_min = [frame.process_angles() for frame in frames_min]

        # trasformo da numpy array a lista
        keypoints_max = [keypoint.tolist() for keypoint in keypoints_max]
        keypoints_min = [keypoint.tolist() for keypoint in keypoints_min]
        angles_max = [angle.tolist() for angle in angles_max]
        angles_min = [angle.tolist() for angle in angles_min]

        return {
            'keypoints_max': keypoints_max,
            'keypoints_min': keypoints_min,
            'angles_max': angles_max,
            'angles_min': angles_min
        }


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