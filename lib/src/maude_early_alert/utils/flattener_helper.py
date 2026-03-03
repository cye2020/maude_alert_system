import datetime
from typing import Any, List, Union, Dict

# ============================================
# MAUDE EVENT 스키마 정의
# ============================================

SCALAR_FIELDS = [
    'adverse_event_flag',
    'date_added',
    'date_changed',
    'date_of_event',
    'date_received',
    'date_report',
    'distributor_address_1',
    'distributor_address_2',
    'distributor_city',
    'distributor_name',
    'distributor_state',
    'distributor_zip_code',
    'distributor_zip_code_ext',
    'event_key',
    'event_location',
    'event_type',
    'exemption_number',
    'health_professional',
    'initial_report_to_fda',
    'manufacturer_address_1',
    'manufacturer_address_2',
    'manufacturer_city',
    'manufacturer_contact_address_1',
    'manufacturer_contact_address_2',
    'manufacturer_contact_area_code',
    'manufacturer_contact_city',
    'manufacturer_contact_country',
    'manufacturer_contact_exchange',
    'manufacturer_contact_extension',
    'manufacturer_contact_f_name',
    'manufacturer_contact_l_name',
    'manufacturer_contact_pcity',
    'manufacturer_contact_pcountry',
    'manufacturer_contact_phone_number',
    'manufacturer_contact_plocal',
    'manufacturer_contact_postal_code',
    'manufacturer_contact_state',
    'manufacturer_contact_t_name',
    'manufacturer_contact_zip_code',
    'manufacturer_contact_zip_ext',
    'manufacturer_country',
    'manufacturer_g1_address_1',
    'manufacturer_g1_address_2',
    'manufacturer_g1_city',
    'manufacturer_g1_country',
    'manufacturer_g1_name',
    'manufacturer_g1_postal_code',
    'manufacturer_g1_state',
    'manufacturer_g1_zip_code',
    'manufacturer_g1_zip_code_ext',
    'manufacturer_link_flag',
    'manufacturer_name',
    'manufacturer_postal_code',
    'manufacturer_state',
    'manufacturer_zip_code',
    'manufacturer_zip_code_ext',
    'mdr_report_key',
    'mfr_report_type',
    'noe_summarized',
    'number_devices_in_event',
    'number_patients_in_event',
    'pma_pmn_number',
    'previous_use_code',
    'product_problem_flag',
    'removal_correction_number',
    'report_number',
    'report_source_code',
    'report_to_fda',
    'report_to_manufacturer',
    'reporter_country_code',
    'reporter_occupation_code',
    'reporter_state_code',
    'reprocessed_and_reused_flag',
    'single_use_flag',
    'summary_report_flag',
    'suppl_dates_fda_received',
    'suppl_dates_mfr_received',
]

ARRAY_FIELDS = [
    'product_problems',
    'remedial_action',
    'source_type',
    'type_of_report',
]

PATIENT_FIELDS = [
    'date_received',
    'patient_age',
    'patient_ethnicity',
    'patient_race',
    'patient_sequence_number',
    'patient_sex',
    'patient_weight',
    'patient_problems',
    'sequence_number_outcome',
    'sequence_number_treatment',
]

MDR_TEXT_FIELDS = [
    'mdr_text_key',
    'patient_sequence_number',
    'text',
    'text_type_code',
]

DEVICE_FIELDS = [
    'brand_name',
    'catalog_number',
    'combination_product_flag',
    'date_received',
    'date_removed_flag',
    'date_removed_year',
    'date_returned_to_manufacturer',
    'device_age_text',
    'device_availability',
    'device_evaluated_by_manufacturer',
    'device_event_key',
    'device_operator',
    'device_report_product_code',
    'device_sequence_number',
    'generic_name',
    'implant_date_year',
    'implant_flag',
    'lot_number',
    'manufacturer_d_address_1',
    'manufacturer_d_address_2',
    'manufacturer_d_city',
    'manufacturer_d_country',
    'manufacturer_d_name',
    'manufacturer_d_postal_code',
    'manufacturer_d_state',
    'manufacturer_d_zip_code',
    'manufacturer_d_zip_code_ext',
    'model_number',
    'other_id_number',
    'serviced_by_3rd_party_flag',
    'udi_di',
    'udi_public',
    'openfda:device_class',
    'openfda:device_name',
    'openfda:medical_specialty_description',
    'openfda:regulation_number',
]


# ============================================
# 스키마 딕셔너리
# ============================================

MAUDE_EVENT_SCHEMA = {
    'scalar_fields': SCALAR_FIELDS,
    'array_fields': ARRAY_FIELDS,
    'first_only': {
        'patient': PATIENT_FIELDS
    },
    'aggregated_array': {
        'mdr_text': MDR_TEXT_FIELDS
    },
    'row_split_array': {
        'device': DEVICE_FIELDS
    }
}