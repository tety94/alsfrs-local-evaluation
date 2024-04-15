import pandas as pd

def prepare_db(db, tests, MIN_ONSET_FIRST_ALSFRS_MONTHS=24, MAX_AGE=75):
    db = db[db['onset_date'].notna()]

    db['alsfrs_tot'] = db['alsfrs_tot'].astype(int)

    db['visit_date'] = pd.to_datetime(db['visit_date'])
    db['onset_date'] = pd.to_datetime(db['onset_date'])
    db['diagnosis_date'] = pd.to_datetime(db['diagnosis_date'])
    db['birth_date'] = pd.to_datetime(db['birth_date'])

    db['first_alsfrs_visit'] = db.groupby('subject_id')['visit_date'].transform('min')

    db['onset_birth'] = db['onset_date'].dt.year - db['birth_date'].dt.year
    db['diagnosis_birth'] = db['diagnosis_date'].dt.year - db['birth_date'].dt.year

    db['onset_first_alsfrs'] = ((db['first_alsfrs_visit'].dt.year - db['onset_date'].dt.year) * 12 +
                                          db['first_alsfrs_visit'].dt.month - db['onset_date'].dt.month)

    #remove patients with delta onset - first alsfrs > MIN_ONSET_FIRST_ALSFRS_MONTHS
    db = db[db['onset_first_alsfrs'] <= MIN_ONSET_FIRST_ALSFRS_MONTHS]

    #remove patients with max 75 years
    db = db[db['diagnosis_birth'] <= MAX_AGE]

    db['onset_visit_delay_months'] = ((db['visit_date'].dt.year - db['onset_date'].dt.year) * 12 +
                                      db['visit_date'].dt.month - db['onset_date'].dt.month)
    
    #to avoid division by 0
    db.loc[db['onset_visit_delay_months'] == 0, 'onset_visit_delay_months'] = 1

    db['first_alsfrs_visit_delay_months'] = ((db['visit_date'].dt.year - db['first_alsfrs_visit'].dt.year) * 12 +
                                          db['visit_date'].dt.month - db['first_alsfrs_visit'].dt.month)

    db['diagnosis_onset_in_months'] = ((db['diagnosis_date'].dt.year - db['onset_date'].dt.year) * 12 +
                                       db['diagnosis_date'].dt.month - db['onset_date'].dt.month)

    db = db.sort_values(by=['subject_id', 'visit_date'])
    db['counter'] = db.groupby('subject_id').cumcount() + 1

    db['copy_subject_id'] = db['subject_id']

    for test in tests:
        db[test] = 0

    return db