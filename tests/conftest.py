from pathlib import Path

import pytest

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "registry"


def make_omop_db(path: Path) -> Path:
    """A miniature OMOP CDM in DuckDB: 4 people, 5 visits, 5 conditions, 4 labs, one empty table."""
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("""
        CREATE TABLE concept (concept_id INTEGER, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR,
                              concept_code VARCHAR);
        INSERT INTO concept VALUES (8507, 'MALE', 'Gender', 'Gender', 'M'), (8532, 'FEMALE', 'Gender', 'Gender', 'F'),
            (9201, 'Inpatient Visit', 'Visit', 'Visit', 'IP'), (9202, 'Outpatient Visit', 'Visit', 'Visit', 'OP'),
            (201826, 'Type 2 diabetes mellitus', 'Condition', 'SNOMED', '44054006'),
            (320128, 'Essential hypertension', 'Condition', 'SNOMED', '59621000'),
            (3004410, 'Hemoglobin A1c/Hemoglobin.total in Blood', 'Measurement', 'LOINC', '4548-4');
        CREATE TABLE person (person_id INTEGER, gender_concept_id INTEGER, year_of_birth INTEGER,
                             person_source_value VARCHAR);
        INSERT INTO person VALUES (1, 8507, 1950, 'SYN-000001'), (2, 8532, 1980, 'SYN-000002'),
            (3, 8532, 2001, 'SYN-000003'), (4, 8507, 1965, 'SYN-000004');
        CREATE TABLE visit_occurrence (visit_occurrence_id INTEGER, person_id INTEGER, visit_concept_id INTEGER,
                                       visit_start_date DATE, visit_end_date DATE, visit_source_value VARCHAR);
        INSERT INTO visit_occurrence VALUES (1, 1, 9202, '2021-03-01', '2021-03-01', 'IME'),
            (2, 2, 9201, '2022-05-02', '2022-05-09', 'IMC'), (3, 2, 9202, '2022-07-07', '2022-07-07', 'IME'),
            (4, 3, 9202, '2023-01-10', '2023-01-10', 'PED'), (5, 4, 9201, '2023-02-01', '2023-02-04', 'IMC');
        CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER, person_id INTEGER,
                                           condition_concept_id INTEGER, condition_start_date DATE,
                                           visit_occurrence_id INTEGER, condition_source_value VARCHAR);
        INSERT INTO condition_occurrence VALUES (10, 1, 201826, '2021-03-01', 1, 'E11.9'),
            (11, 2, 201826, '2022-05-02', 2, 'E11.9'), (12, 2, 320128, '2022-07-07', 3, 'I10.9'),
            (13, 4, 320128, '2023-02-01', 5, 'I10.9'), (14, 4, 201826, '2023-02-01', 5, 'E11.9');
        CREATE TABLE measurement (measurement_id INTEGER, person_id INTEGER, measurement_concept_id INTEGER,
                                  measurement_date DATE, value_as_number DOUBLE, unit_source_value VARCHAR);
        INSERT INTO measurement VALUES (100, 1, 3004410, '2021-03-01', 7.1, '%'), (101, 2, 3004410, '2022-05-02', 6.4, '%'),
            (102, 2, 3004410, '2022-07-07', NULL, '%'), (103, 4, 3004410, '2023-02-01', 8.25, '%');
        CREATE TABLE specimen (specimen_id INTEGER, person_id INTEGER);
    """)
    con.close()
    return path


@pytest.fixture
def omop_db(tmp_path) -> Path:
    return make_omop_db(tmp_path / "omop.duckdb")


@pytest.fixture
def tmp_session():
    from medbench.db import init_db, make_engine, session_factory

    engine = make_engine("sqlite://")
    init_db(engine)
    session = session_factory(engine)()
    yield session
    session.close()
