from graphviz import Digraph

dot = Digraph()

dot.attr(size='20,20')
dot.attr(dpi='600')

dot.node('Input1', 'Input Layer 1\n(input_size_1)')
dot.node('LSTM1_1', 'LSTM 1-1\n(hidden_size_1)')
dot.node('Dropout1_1', 'Dropout 1-1\n(dropout_rate)')
dot.node('LSTM2_1', 'LSTM 2-1\n(hidden_size_2)')
dot.node('Dropout2_1', 'Dropout 2-1\n(dropout_rate)')

dot.node('Input2', 'Input Layer 2\n(input_size_2)')
dot.node('LSTM1_2', 'LSTM 1-2\n(hidden_size_1)')
dot.node('Dropout1_2', 'Dropout 1-2\n(dropout_rate)')
dot.node('LSTM2_2', 'LSTM 2-2\n(hidden_size_2)')
dot.node('Dropout2_2', 'Dropout 2-2\n(dropout_rate)')

dot.node('Input3', 'Input Layer 3\n(input_size_3)')
dot.node('LSTM1_3', 'LSTM 1-3\n(hidden_size_1)')
dot.node('Dropout1_3', 'Dropout 1-3\n(dropout_rate)')
dot.node('LSTM2_3', 'LSTM 2-3\n(hidden_size_2)')
dot.node('Dropout2_3', 'Dropout 2-3\n(dropout_rate)')

# Aggiungi nodi per i layer fully connected e ReLU
dot.node('FC1', 'Dense 1\n(hidden_size_3)')
dot.node('ReLU', 'ReLU')
dot.node('Dropout_ReLU', 'Dropout 3\n(dropout_rate)')  # Nuovo nodo per Dropout dopo ReLU
dot.node('FC2', 'Dense 2\n(num_classes)')
dot.node('Sigmoid', 'Sigmoid')

# Aggiungi le connessioni
# Ramo 1
dot.edge('Input1', 'LSTM1_1')
dot.edge('LSTM1_1', 'Dropout1_1')
dot.edge('Dropout1_1', 'LSTM2_1')
dot.edge('LSTM2_1', 'Dropout2_1')

# Ramo 2
dot.edge('Input2', 'LSTM1_2')
dot.edge('LSTM1_2', 'Dropout1_2')
dot.edge('Dropout1_2', 'LSTM2_2')
dot.edge('LSTM2_2', 'Dropout2_2')

# Ramo 3
dot.edge('Input3', 'LSTM1_3')
dot.edge('LSTM1_3', 'Dropout1_3')
dot.edge('Dropout1_3', 'LSTM2_3')
dot.edge('LSTM2_3', 'Dropout2_3')

# Connessioni ai layer fully connected
dot.edge('Dropout2_1', 'FC1')
dot.edge('Dropout2_2', 'FC1')
dot.edge('Dropout2_3', 'FC1')
dot.edge('FC1', 'ReLU')
dot.edge('ReLU', 'Dropout_ReLU')  # Connessione verso il nuovo Dropout dopo ReLU
dot.edge('Dropout_ReLU', 'FC2')  # Connessione da Dropout 3 a FC2
dot.edge('FC2', 'Sigmoid')

# Salva come immagine
dot.render('lstm_model_structure', format='png', cleanup=True)
