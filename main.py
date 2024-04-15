import os
import matplotlib.pyplot as plt
from decreasing_functions import *
from db_cleaning import *

db = pd.read_csv('alsfrs.csv', delimiter=';', encoding='utf-8', decimal=',', low_memory=False)

PLOT = True
DECIMAL_ROUNDS = 0
MONTHS_DELAY = 1
IMAGES_FOLDER = 'images'
MONTHS = [6,12]

result_folder = 'fractional_tests/'
if DECIMAL_ROUNDS == 0:
    result_folder = 'integer_tests/'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

tests = ['d50_revised','rational', 'delta', 'ln']

db = prepare_db(db, tests).copy(deep=True)

def save_plots(group, tests):
    x_continuos = np.arange(0,60, 0.1)
    first_row = group.iloc[0]
    code = first_row['subject_id']

    for test in tests:
        match test:
            case 'd50_revised':
                y_data = d50_revised_function(x_continuos, first_row['d50_revised'])
                color = 'g'
            case 'rational':
                y_data = rational_function(x_continuos, first_row['rational'])
                color = 'k'
            case 'delta':
                y_data = delta_function(x_continuos, first_row['delta'])
                color = 'r'
            case 'ln':
                y_data = ln_function(x_continuos, first_row['ln'])
                color = 'y'

        plt.plot(x_continuos, np.round(y_data,DECIMAL_ROUNDS) ,color=f'{color}', linestyle='-')

    #raw data
    plt.ylim([0, 49])
    plt.axvline(x=first_row['diagnosis_onset_in_months'])

    #plot (0,48) and first alsfrs point
    onset_and_first_visit_x = [0, group[group['counter'] == 1]['onset_visit_delay_months'].iloc[0]]
    onset_and_first_visit_y = [48, group[group['counter'] == 1]['alsfrs_tot'].iloc[0]]
    plt.scatter(onset_and_first_visit_x, onset_and_first_visit_y, color='r', marker='o', linewidths=5)

    #plot other visits
    other_visits_x = group[group['counter'] > 2]['onset_visit_delay_months'].values
    other_visits_y = group[group['counter'] > 2]['alsfrs_tot'].values
    plt.scatter(other_visits_x, other_visits_y, color='g', marker='o', linewidths=5)

    plt.xlabel('Time (months)')
    plt.ylabel('ALSFRS_R - TOT')
    plt.savefig(f"{IMAGES_FOLDER}/{code}.png")
    plt.clf()


def add_values_to_group(group):
    for test in tests:
        x_data = group.iloc[0]['onset_visit_delay_months']
        y_data = group.iloc[0]['alsfrs_tot']
        parameter = calculate_parameter(test + '_calculate_parameter_function', x_data, y_data)
        group[test] = parameter

    if PLOT:
        save_plots(group, tests)

    return group


print('### START OF ALGORITHM ###')
db = db.groupby('subject_id').apply(add_values_to_group)
db = db.reset_index(drop=True)
print('### END OF MODELS CALCULATION ###')

for test in tests:
    y_data = calculate(f'{test}_function', db['onset_visit_delay_months'].values, db[test].values)
    db['alsfrs_tot_predicted_by_' + test] = np.maximum(np.round(y_data, DECIMAL_ROUNDS), 0)
    db['diff_tot_' + test] = abs(db['alsfrs_tot'] - db['alsfrs_tot_predicted_by_' + test])


for month in MONTHS:
    col = f'sort_{month}_months'
    db[col] = 1000

    #we use only one alsfrs, the closest to the month, with priority of the next month
    db.loc[db['first_alsfrs_visit_delay_months'] == month, col] = 1
    db.loc[db['first_alsfrs_visit_delay_months'].between(month, month + MONTHS_DELAY), col] = 2
    db.loc[db['first_alsfrs_visit_delay_months'].between(month - MONTHS_DELAY, month), col] = 3

    db_months = db.sort_values(by=col).groupby('subject_id').first().reset_index()
    db_months = db_months[db_months[col] < 4]

    result_list = ['diff_tot_' + item  for item in tests]

    #calculate final results
    results_models_model = []
    results_models_patients_months = []
    results_models_percentile = []

    min_val = db_months[result_list].min(axis=1)

    for test in tests:
        db_months[f'is_{test}_best'] = 0
        db_months.loc[db_months['diff_tot_' + test] == min_val, f'is_{test}_best'] = 1
        results_models_model.append(test)
        results_models_patients_months.append(len(db_months))

        perc_25 = np.percentile(db_months['diff_tot_' + test].dropna(), 25)
        perc_50 = np.percentile(db_months['diff_tot_' + test].dropna(), 50)
        perc_75 = np.percentile(db_months['diff_tot_' + test].dropna(), 75)

        results_models_percentile.append(f"{perc_50} ({perc_25} - {perc_75})")

    dict = {
        'models' : results_models_model,
        'patients' : results_models_patients_months,
        'percentile' : results_models_percentile,
    }
    model_results_df = pd.DataFrame(dict)
    model_results_df.to_csv(f'{result_folder}model_results_{month}_df.csv', decimal=',', sep=';')
    db_months.to_csv(f'{result_folder}results_{month}_months.csv', decimal=',', sep=';')

db.to_csv(f'{result_folder}alsfrs_modified.csv', decimal=',', sep=';')

print('### END OF ALGORITHM ###')
