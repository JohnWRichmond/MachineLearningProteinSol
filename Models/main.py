import run_pycaret

config = {'datasets': ({'niwa': 'C:/Users/johnr/Documents/Year2/GNN/MachineLearningProteinSol/Processed_data/Solubility/niwa.csv',
                        'AS500': 'C:/Users/johnr/Documents/Year2/GNN/MachineLearningProteinSol/Processed_data/Solubility/AS500.csv',
                        'AS1000': 'C:/Users/johnr/Documents/Year2/GNN/MachineLearningProteinSol/Processed_data/Solubility/AS1000.csv',
                        'AS2000': 'C:/Users/johnr/Documents/Year2/GNN/MachineLearningProteinSol/Processed_data/Solubility/AS2000.csv'}),

          'machine_learning_algorithms': ('lr', 'rf', 'et', 'ard', 'gbr'),

          'target': ({'niwa': 'log_solubility',
                      'AS500': 'log_solubility',
                      'AS1000': 'log_solubility',
                      'AS2000': 'log_solubility'})
          }

run_pycaret.experiments(config)
