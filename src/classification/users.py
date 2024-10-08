import util
import os

class Users:
    """
    Classe che gestisce utenti e sessione corrente
    """

    def __init__(self):
        """
        Costruttore della classe
        """

        self.current_user = None
        self.current_session = None
        self.credentials_path = os.path.join(util.getUsersPath(), "credentials.json")
        self.data_path = os.path.join(util.getUsersPath(), "data.json")


    def user_exists(self, username):
        """
        Funzione che verifica se l'utente esiste già

        Args:
        - username (string): username dell'utente

        Returns:
        - (bool): vero se l'utente esiste, falso altrimenti
        """

        credentials = util.read_json(self.credentials_path)
        return username in [credentials[i]["username"] for i in range(len(credentials))]
    

    def add_user(self, username, password):
        """
        Funzione che aggiunge un utente

        Args:
        - username (string): username dell'utente
        - password (string): password dell'utente
        """

        credentials = util.read_json(self.credentials_path)
        credentials.append({"username": username, "password": password})
        util.write_json(credentials, self.credentials_path)

    
    def login(self, username, password):
        """
        Funzione che effettua il login e salva l'utente e la sessione corrente

        Args:
        - username (string): username dell'utente
        - password (string): password dell'utente

        Returns:
        - (bool): vero se il login è avvenuto con successo, falso altrimenti
        """

        if not self.user_exists(username):
            self.add_user(username, password)
        elif not password == util.read_json(self.credentials_path)[username]["password"]:
            return False

        date = util.get_current_date()
        self.current_user = username
        self.current_session = date
        return True


    def update_session(self, exercise, reps, accuracy, avg_time):
        """
        Funzione che aggiorna i dati relativi all'esercizio eseguito

        Args:
        - exercise (string): esercizio eseguito
        - reps (int): numero di ripetizioni
        - accuracy (float): accuratezza
        - avg_time (float): tempo medio
        """

        data = util.read_json(self.data_path)
        if self.current_user not in data.keys():
            data[self.current_user] = {}
        if self.current_session not in data[self.current_user].keys():
            data[self.current_user][self.current_session] = {}
        data[self.current_user][self.current_session][exercise] = {}
        data[self.current_user][self.current_session][exercise]["reps"] = reps
        data[self.current_user][self.current_session][exercise]["accuracy"] = accuracy
        data[self.current_user][self.current_session][exercise]["avg_time"] = avg_time