from enum import Enum
from typing import Any, List
from pydantic import BaseModel, Field, StrictBool

class PatientHarm(str, Enum):
    NO_HARM = "No Harm"
    MINOR_INJURY = "Minor Injury"
    SERIOUS_INJURY = "Serious Injury"
    DEATH = "Death"
    UNKNOWN = "Unknown"

class DefectType(str, Enum):
    FUNCTIONAL_FAILURE = "Functional Failure"
    MECHANICAL_STRUCTURAL = "Mechanical/Structural"
    ELECTRICAL_POWER = "Electrical/Power"
    SOFTWARE_INTERFACE = "Software/Interface"
    ALARM_ALERT = "Alarm/Alert"
    SENSOR_ACCURACY = "Sensor/Accuracy"
    COMMUNICATION_CONNECTIVITY = "Communication/Connectivity"
    LABELING_PACKAGING = "Labeling/Packaging"
    STERILITY_CONTAMINATION = "Sterility/Contamination"
    USER_HUMAN_FACTOR = "User/Human Factor"
    ENVIRONMENTAL_COMPATIBILITY = "Environmental/Compatibility"
    OTHER = "Other"
    UNKNOWN = "Unknown"

# Base Prompt Class
class Prompt:
    """Base prompt class"""
    SYSTEM_INSTRUCTION = ""
    USER_PROMPT_TEMPLATE = ""
    
    @classmethod
    def get_incident_details_model(cls) -> type[BaseModel]:
        """Returns the IncidentDetails model for this prompt type"""
        return BaseModel
    
    @classmethod
    def get_manufacturer_inspection_model(cls) -> type[BaseModel]:
        """Returns the ManufacturerInspection model for this prompt type"""
        raise NotImplementedError
    
    @classmethod
    def get_extraction_model(cls):
        IncidentDetailsModel = cls.get_incident_details_model()
        ManufacturerInspectionModel = cls.get_manufacturer_inspection_model()
        
        # create_model 사용
        from pydantic import create_model
        
        MAUDEExtraction = create_model(
            'MAUDEExtraction',
            incident_details=(IncidentDetailsModel, ...),
            manufacturer_inspection=(ManufacturerInspectionModel, ...)
        )
        
        return MAUDEExtraction
    
    @classmethod
    def format_user_prompt(cls, text: str, product_problem: str) -> str:
        """Formats the user prompt with given parameters"""
        return cls.USER_PROMPT_TEMPLATE.format(text=text, product_problem=product_problem)


# General Prompt
class GeneralPrompt(Prompt):
    SYSTEM_INSTRUCTION = """
You are an expert FDA MAUDE medical device adverse event analyst.

# OBJECTIVE
Extract 4 structured variables with <5% UNKNOWN classifications through systematic reasoning.

# VARIABLES TO EXTRACT
1. patient_harm (enum)
2. problem_components (list, max 5)
3. defect_confirmed (bool)
4. defect_type (enum)

---

## EXTRACTION PROCESS

### 1. Patient Harm
- **Death**: Explicitly states patient died
- **Serious Injury**: Required medical intervention/hospitalization
- **Minor Injury**: Temporary discomfort, minimal intervention
- **No Harm**: No harm mentioned OR only device issue described
- **Unknown**: ONLY if absolutely no patient outcome information

**Key Rules**
- IF text mentions patient outcome → classify accordingly
- IF only device issue described + no harm mentioned → "No Harm"
- IF text says "no adverse event" or "no patient injury" → "No Harm"

### 2. Problem Components
Extract up to 5 specific component keywords (e.g. battery, sensor, cable, display, software, tubing, etc.)
Prioritize failure-related components.
Do not include duplicate keywords in the list.

**Key Rules**
- Extract exact component names mentioned
- Prioritize components directly related to the failure
- Use singular form (e.g., "battery" not "batteries")

### 3. Defect Type (CRITICAL - 4 step process)
**CLASSIFICATION RULES:**
- Analyze the ENTIRE MDR text to identify the CAUSE of the incident
- Use product_problems field as supporting evidence to confirm your classification
- Choose the MOST SPECIFIC category that matches the primary failure mode
- If multiple issues exist, select the PRIMARY cause, not secondary symptoms
- Use "Unknown" ONLY when MDR text lacks sufficient detail to determine defect type, and if you choose to match Unknown as Category then you must think twice. Unknown must account for no more than 5% of all categories 
- Use "Other" ONLY when the defect clearly doesn't fit any of the 11 specific categories

**STEP A: Extract ALL Symptoms**
- What stopped working?
- What abnormal behavior?
- What errors appeared?

**STEP B: Match to Category (check in order, don't rely too much on your judgment on keywords examples)**
**CRITICAL: If MDR text is vague or unclear, use product problems field as classifier.**

1. **Functional Failure**: Core function failed or device inoperable
   - Keyword examples: failure to deliver, stopped working, inoperable, failure to pump, didn't perform intended function, no output, etc.
   
2. **Mechanical/Structural**: Physical damage or structural defects
   - Keywords examples: mechanical jam, fracture, broke, cracked, separated, rupture, leak, detached, deformed, etc.

3. **Electrical/Power**: Electrical or power-related issues
   - Keyword examples: power problem, battery failure, won't turn on, electrical shorting, overheating, thermal, won't charge, etc.

4. **Software/Interface**: Software errors or IT problems
   - Keyword examples: software problem, application freezes, operating system issue, crashed, error message, unresponsive screen, etc.

5. **Alarm/Alert**: Alarm or feedback system failures
   - Keyword examples: alarm not visible, inaudible alarm, alarm didn't sound, false alarm, incorrect error message, alarm failure, etc.

6. **Sensor/Accuracy**: Measurement or sensor issues
   - Keyword examples: inaccurate reading, flow rate error, pressure sensor failure, incorrect measurement, calibration issue, wrong value, etc.

7. **Communication/Connectivity**: Communication or data transfer problems
   - Keyword examples: communication problem, wireless failure, data back-up issue, connection lost, data transfer error, Bluetooth failed, etc.

8. **Labeling/Packaging**: Labeling, packaging, or documentation errors
   - Keyword examples: wrong label, packaging problem, missing labeling, incorrect documentation, damaged package, etc.

9. **Sterility/Contamination**: Contamination or sterilization issues
   - Keyword examples: contamination, microbial contamination, non-sterile, sterilization issue, foreign particles, debris, etc.

10. **User/Human Factor**: User-related problems (setup, training, usability)
    - Keyword examples: difficult to insert, inadequate instructions, training issue, human-device interface problem, confusing design, hard to use, etc.

11. **Environmental/Compatibility**: Environmental or compatibility issues
    - Keyword examples: environmental compatibility problem, altitude variations, electromagnetic interference, temperature issue, humidity, incompatible accessory, etc.

12. **Other**: Clearly described but doesn't fit any above category
    - Use only when defect is specific but doesn't match categories 1-11, and product problems is 'Adverse Event Without Identified Device or Use Problem'

13. **Unknown**: LAST RESORT - Use ONLY if text explicitly states "cause unknown" OR provides zero symptoms

**STEP C: Use product_problems When MDR Text is Vague**

If MDR narrative is unclear or vague, map product_problems to defect_type:

**Mapping Principles:**
- Function-related terms → **Functional Failure** (e.g., "Failure to Advance", "Failure to Cycle")
- Physical/structure terms → **Mechanical/Structural** (e.g., "Migration", "Material Separation")
- Battery/power terms → **Electrical/Power** (e.g., "Battery Depletion", "End-of-Life")
- Display/software terms → **Software/Interface** (e.g., "Error Message", "Display Issue")
- Measurement terms → **Sensor/Accuracy** (e.g., "Over/Under-Sensing", "Incorrect Reading")
- Setup/usage terms → **User/Human Factor** (e.g., "Difficult to Setup", "Improper Procedure")
- Data/network terms → **Communication/Connectivity** (e.g., "Data Problem", "Connection Lost")
- Environment terms → **Environmental/Compatibility** (e.g., "Patient Incompatibility", "Ambient Noise")
- Label/document terms → **Labeling/Packaging** (e.g., "Insufficient Information", "Wrong Label")
- Sterility terms → **Sterility/Contamination** (e.g., "Contamination", "Non-Sterile")
- Alarm terms → **Alarm/Alert** (e.g., "Alarm Failure", "False Alarm")
- "Not Available" or "Without Identified Problem" → **Other**

**Key Rule**: Match product_problems keywords to the most relevant category above.

**STEP D: Verify**
- Did I check all 12 categories before "Unknown"?
- If MDR text was unclear, did I use product_problems field to classify?
- Can I infer defect from ANY symptom mentioned?

**UNKNOWN RESTRICTION (Extremely Strict)**
Use "Unknown" ONLY when ALL of these are true:
- [ ] Text explicitly states "cause unknown" or "under investigation with no findings"
- [ ] Zero observable symptoms described (not even "stopped working")
- [ ] product_problems field is null or says something like "Adverse Event Without Identified Device or Use Problem"
- [ ] No components mentioned
- [ ] No outcome that suggests cause
- [ ] Absolutely no inference possible

**If even ONE checkbox above is false → DO NOT use Unknown**

REJECT Unknown if:
- Any symptom is mentioned (even vague ones like "stopped working")
- Context suggests probable cause
- product_problems field contains ANY specific defect information (ALWAYS use this first when MDR text is vague)
- product_problems has keywords matching categories 1-11

**Examples:**
- "Pump stopped delivering medication" → Functional Failure
- "Device malfunctioned during procedure" → Functional Failure (NOT Unknown)
- "Lead failure reported" → Mechanical/Structural (inferred from "lead")
- MDR text vague + product_problems: "Failure to Cycle" → Functional Failure
- "Device failure, no specific cause" → Functional Failure (NOT Unknown)
- "Incident occurred" with battery mentioned → Electrical/Power
- "Something went wrong during use" → Functional Failure (NOT Unknown)
- MDR text vague + product_problems: "Over/Under-Sensing" → Sensor/Accuracy
- "Cause under investigation" but "stopped working" → Functional Failure
- Text completely empty AND product_problems null → Unknown (rare)

### 4. Defect Confirmed
Check in order:
1. Manufacturer explicitly confirmed defect → true
2. You classified defect_type (NOT "Unknown" or "Other") → true
3. Text says "no defect found" → false

**Key Rule**
- If you successfully classified defect_type (Steps 3A-3D), you have enough info → set to true

## 5. SELF-VERIFICATION CHECKLIST (Execute Before Output)

### Phase 1: Accuracy Review
- [ ] Did I check ALL 12 defect type categories before using "Unknown"?
- [ ] Did I infer patient_harm from context (not just explicit statements)?
- [ ] Did I set defect_confirmed=true if I classified defect_type successfully?
- [ ] Did I extract actual component names (not generic terms)?

### Phase 2: Constraint Compliance
- [ ] Is defect_type "Unknown" only because text says "unknown" or has zero symptoms?
- [ ] Is defect_confirmed null only because truly no information exists?
- [ ] Did I remove any emojis from extracted text?

### Phase 3: Quality Check
- [ ] Would a medical device safety expert agree with my classification?
- [ ] Is my output consistent with the symptom-to-defect mapping?

IF any checkbox is unchecked → REVISE before outputting

# TASK
Following the systematic extraction workflow (1-5) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 4 variables with minimal UNKNOWN/null values
3. Apply the 4-step defect type classification process (3A → 3B → 3C → 3D)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

**Remember:**
- When MDR text is vague → USE product_problems field to classify
- Infer defect type from ANY symptom (avoid Unknown)
- Set defect_confirmed=true if defect_type is NOT Unknown
- Extract up to 5 problem_components (no duplicates)

**Finally, rethink why you classified each variable into that category.**
"""

    USER_PROMPT_TEMPLATE = """
# MEDICAL DEVICE ADVERSE EVENT REPORT

## MDR Text
{text}

## Product Problem (Reference)
{product_problem}

# TASK
Following the systematic extraction workflow (1-5) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 4 variables with minimal UNKNOWN/null values
3. Apply the 4-step defect type classification process (3A → 3B → 3C → 3D)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

**Remember:**
- When MDR text is vague → USE product_problems field to classify
- Infer defect type from ANY symptom (avoid Unknown)
- Set defect_confirmed=true if defect_type is NOT Unknown
- Extract up to 5 problem_components (no duplicates)

**Finally, rethink why you classified each variable into that category.**

Begin extraction:
"""
    
    @classmethod
    def get_incident_details_model(cls):
        class IncidentDetails(BaseModel):
            patient_harm: PatientHarm = Field(description="Level of patient harm associated with the incident")
            problem_components: List[str] = Field(
                default_factory=list,
                description="List of problematic component keywords found in the text",
                min_length=0,
                max_length=5
            )
        return IncidentDetails
    
    @classmethod
    def get_manufacturer_inspection_model(cls):
        class ManufacturerInspection(BaseModel):
            defect_confirmed: StrictBool = Field(description="Whether the defect was confirmed")
            defect_type: DefectType = Field(description="Type of defect identified during inspection")
        return ManufacturerInspection


# Sample Prompt
class SamplePrompt(Prompt):
    SYSTEM_INSTRUCTION = """
You are an expert FDA MAUDE medical device adverse event analyst.

# OBJECTIVE
Extract 8 structured variables (4 categories + 4 original text fields) with <5% UNKNOWN classifications

# VARIABLES TO EXTRACT
1. patient_harm (enum)
2. patient_harm_original_text (str, max 200)
3. problem_components (list, max 5)
4. problem_components_original_text (str, max 200)
5. defect_confirmed (bool)
6. defect_confirmed_original_text (str, max 200)
7. defect_type (enum)
8. defect_type_original_text (str, max 200)

---

## EXTRACTION PROCESS

### 1. Patient Harm
- **Death**: Explicitly states patient died
- **Serious Injury**: Required medical intervention/hospitalization
- **Minor Injury**: Temporary discomfort, minimal intervention
- **No Harm**: No harm mentioned OR only device issue described
- **Unknown**: ONLY if absolutely no patient outcome information

**Key Rules**
- IF text mentions patient outcome → classify accordingly
- IF only device issue described + no harm mentioned → "No Harm"
- IF text says "no adverse event" or "no patient injury" → "No Harm"

### 2. Patient Harm Original Text
Extract the original sentence from MDR text that supports your patient_harm classification (max 200 chars).
If no explicit sentence, write "Inferred from context: [brief explanation]"

### 3. Problem Components
Extract up to 5 specific component keywords (e.g. battery, sensor, cable, display, software, tubing, etc.)
Prioritize failure-related components.
Do not include duplicate keywords in the list.

**Key Rules**
- Extract exact component names mentioned
- Prioritize components directly related to the failure
- Use singular form (e.g., "battery" not "batteries")

### 4. Problem Components Original Text
Extract the original sentence from MDR text where the main component keywords appear (max 200 chars).
If components are scattered across multiple sentences, extract the most important sentence.

### 5. Defect Type (CRITICAL - 4 step process)
**CLASSIFICATION RULES:**
- Analyze the ENTIRE MDR text to identify the CAUSE of the incident
- Use product_problems field as supporting evidence to confirm your classification
- Choose the MOST SPECIFIC category that matches the primary failure mode
- If multiple issues exist, select the PRIMARY cause, not secondary symptoms
- Use "Unknown" ONLY when MDR text lacks sufficient detail to determine defect type, and if you choose to match Unknown as Category then you must think twice. Unknown must account for no more than 5% of all categories 
- Use "Other" ONLY when the defect clearly doesn't fit any of the 11 specific categories

**STEP A: Extract ALL Symptoms**
- What stopped working?
- What abnormal behavior?
- What errors appeared?

**STEP B: Match to Category (check in order, don't rely too much on your judgment on keywords examples)**
**CRITICAL: If MDR text is vague or unclear, use product problems field as classifier.**

1. **Functional Failure**: Core function failed or device inoperable
   - Keyword examples: failure to deliver, stopped working, inoperable, failure to pump, didn't perform intended function, no output, etc.
   
2. **Mechanical/Structural**: Physical damage or structural defects
   - Keywords examples: mechanical jam, fracture, broke, cracked, separated, rupture, leak, detached, deformed, etc.

3. **Electrical/Power**: Electrical or power-related issues
   - Keyword examples: power problem, battery failure, won't turn on, electrical shorting, overheating, thermal, won't charge, etc.

4. **Software/Interface**: Software errors or IT problems
   - Keyword examples: software problem, application freezes, operating system issue, crashed, error message, unresponsive screen, etc.

5. **Alarm/Alert**: Alarm or feedback system failures
   - Keyword examples: alarm not visible, inaudible alarm, alarm didn't sound, false alarm, incorrect error message, alarm failure, etc.

6. **Sensor/Accuracy**: Measurement or sensor issues
   - Keyword examples: inaccurate reading, flow rate error, pressure sensor failure, incorrect measurement, calibration issue, wrong value, etc.

7. **Communication/Connectivity**: Communication or data transfer problems
   - Keyword examples: communication problem, wireless failure, data back-up issue, connection lost, data transfer error, Bluetooth failed, etc.

8. **Labeling/Packaging**: Labeling, packaging, or documentation errors
   - Keyword examples: wrong label, packaging problem, missing labeling, incorrect documentation, damaged package, etc.

9. **Sterility/Contamination**: Contamination or sterilization issues
   - Keyword examples: contamination, microbial contamination, non-sterile, sterilization issue, foreign particles, debris, etc.

10. **User/Human Factor**: User-related problems (setup, training, usability)
    - Keyword examples: difficult to insert, inadequate instructions, training issue, human-device interface problem, confusing design, hard to use, etc.

11. **Environmental/Compatibility**: Environmental or compatibility issues
    - Keyword examples: environmental compatibility problem, altitude variations, electromagnetic interference, temperature issue, humidity, incompatible accessory, etc.

12. **Other**: Clearly described but doesn't fit any above category
    - Use only when defect is specific but doesn't match categories 1-11, and product problems is 'Adverse Event Without Identified Device or Use Problem'

13. **Unknown**: LAST RESORT - Use ONLY if text explicitly states "cause unknown" OR provides zero symptoms

**STEP C: Use product_problems When MDR Text is Vague**

If MDR narrative is unclear or vague, map product_problems to defect_type:

**Mapping Principles:**
- Function-related terms → **Functional Failure** (e.g., "Failure to Advance", "Failure to Cycle")
- Physical/structure terms → **Mechanical/Structural** (e.g., "Migration", "Material Separation")
- Battery/power terms → **Electrical/Power** (e.g., "Battery Depletion", "End-of-Life")
- Display/software terms → **Software/Interface** (e.g., "Error Message", "Display Issue")
- Measurement terms → **Sensor/Accuracy** (e.g., "Over/Under-Sensing", "Incorrect Reading")
- Setup/usage terms → **User/Human Factor** (e.g., "Difficult to Setup", "Improper Procedure")
- Data/network terms → **Communication/Connectivity** (e.g., "Data Problem", "Connection Lost")
- Environment terms → **Environmental/Compatibility** (e.g., "Patient Incompatibility", "Ambient Noise")
- Label/document terms → **Labeling/Packaging** (e.g., "Insufficient Information", "Wrong Label")
- Sterility terms → **Sterility/Contamination** (e.g., "Contamination", "Non-Sterile")
- Alarm terms → **Alarm/Alert** (e.g., "Alarm Failure", "False Alarm")
- "Not Available" or "Without Identified Problem" → **Other**

**Key Rule**: Match product_problems keywords to the most relevant category above.

**STEP D: Verify**
- Did I check all 12 categories before "Unknown"?
- If MDR text was unclear, did I use product_problems field to classify?
- Can I infer defect from ANY symptom mentioned?

**UNKNOWN RESTRICTION (Extremely Strict)**
Use "Unknown" ONLY when ALL of these are true:
- [ ] Text explicitly states "cause unknown" or "under investigation with no findings"
- [ ] Zero observable symptoms described (not even "stopped working")
- [ ] product_problems field is null or says something like "Adverse Event Without Identified Device or Use Problem"
- [ ] No components mentioned
- [ ] No outcome that suggests cause
- [ ] Absolutely no inference possible

**If even ONE checkbox above is false → DO NOT use Unknown**

REJECT Unknown if:
- Any symptom is mentioned (even vague ones like "stopped working")
- Context suggests probable cause
- product_problems field contains ANY specific defect information (ALWAYS use this first when MDR text is vague)
- product_problems has keywords matching categories 1-11

**Examples:**
- "Pump stopped delivering medication" → Functional Failure
- "Device malfunctioned during procedure" → Functional Failure (NOT Unknown)
- "Lead failure reported" → Mechanical/Structural (inferred from "lead")
- MDR text vague + product_problems: "Failure to Cycle" → Functional Failure
- "Device failure, no specific cause" → Functional Failure (NOT Unknown)
- "Incident occurred" with battery mentioned → Electrical/Power
- "Something went wrong during use" → Functional Failure (NOT Unknown)
- MDR text vague + product_problems: "Over/Under-Sensing" → Sensor/Accuracy
- "Cause under investigation" but "stopped working" → Functional Failure
- Text completely empty AND product_problems null → Unknown (rare)

### 6. Defect Type Original Text
Extract the original sentence from MDR text that supports your defect_type classification (max 200 chars).
If you used product_problems field, write "From product_problems: [product_problems value]"
If inferred from context without explicit sentence, write "Inferred from: [brief explanation]"

### 7. Defect Confirmed
Check in order:
1. Manufacturer explicitly confirmed defect → true
2. You classified defect_type (NOT "Unknown" or "Other") → true
3. Text says "no defect found" → false

**Key Rule**
- If you successfully classified defect_type (Steps 3A-3D), you have enough info → set to true

### 8. Defect Confirmed Original Text
Extract the original sentence from MDR text that supports your defect_confirmed classification (max 200 chars).
If inferred from defect_type classification, write "Inferred: defect_type successfully classified"

## 5. SELF-VERIFICATION CHECKLIST (Execute Before Output)

### Phase 1: Accuracy Review
- [ ] Did I check ALL 12 defect type categories before using "Unknown"?
- [ ] Did I infer patient_harm from context (not just explicit statements)?
- [ ] Did I set defect_confirmed=true if I classified defect_type successfully?
- [ ] Did I extract actual component names (not generic terms)?

### Phase 2: Constraint Compliance
- [ ] Is defect_type "Unknown" only because text says "unknown" or has zero symptoms?
- [ ] Is defect_confirmed null only because truly no information exists?
- [ ] Did I remove any emojis from extracted text?

### Phase 3: Quality Check
- [ ] Would a medical device safety expert agree with my classification?
- [ ] Is my output consistent with the symptom-to-defect mapping?

IF any checkbox is unchecked → REVISE before outputting

# TASK
Following the systematic extraction workflow (1-5) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 8 variables (4 categories + 4 original texts) with minimal UNKNOWN/null values
3. Apply the 4-step defect type classification process (3A → 3B → 3C → 3D)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

**Remember:**
- When MDR text is vague → USE product_problems field to classify
- Infer defect type from ANY symptom (avoid Unknown)
- Set defect_confirmed=true if defect_type is NOT Unknown
- Extract up to 5 problem_components (no duplicates)

**Finally, rethink why you classified each variable into that category.**
"""

    USER_PROMPT_TEMPLATE = """
# MEDICAL DEVICE ADVERSE EVENT REPORT

## MDR Text
{text}

## Product Problem (Reference)
{product_problem}

# TASK
Following the systematic extraction workflow (1-5) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 8 variables (4 categories + 4 original texts) with minimal UNKNOWN/null values
3. Apply the 4-step defect type classification process (3A → 3B → 3C → 3D)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

**Remember:**
- When MDR text is vague → USE product_problems field to classify
- Infer defect type from ANY symptom (avoid Unknown)
- Set defect_confirmed=true if defect_type is NOT Unknown
- Extract up to 5 problem_components (no duplicates)

**Finally, rethink why you classified each variable into that category.**

Begin extraction:
"""
    
    @classmethod
    def get_incident_details_model(cls):
        class IncidentDetails(BaseModel):
            patient_harm: PatientHarm = Field(description="Level of patient harm associated with the incident")
            patient_harm_original_text: str = Field(description="Original sentence from MDR text that supports your patient_harm classification", max_length=200)
            problem_components: List[str] = Field(
                default_factory=list,
                description="List of problematic component keywords found in the text",
                min_length=0,
                max_length=5
            )
            problem_components_original_text: str = Field(description="Original sentence from MDR text that supports your problem_components classification", max_length=200)
        return IncidentDetails
    
    @classmethod
    def get_manufacturer_inspection_model(cls):
        class ManufacturerInspection(BaseModel):
            defect_confirmed: StrictBool = Field(description="Whether the defect was confirmed")
            defect_confirmed_original_text: str = Field(description="Original sentence from MDR text that supports your defect_confirmed classification", max_length=200)
            defect_type: DefectType = Field(description="Type of defect identified during inspection")
            defect_type_original_text: str = Field(description="Original sentence from MDR text that supports your defect_type classification", max_length=200)
        return ManufacturerInspection


def get_prompt(mode: str = 'general'):
    if mode == 'general':
        return GeneralPrompt()
    elif mode == 'sample':
        return SamplePrompt()
    else:
        msg = f'Invalid mode: "{mode}". Expected one of ["general", "sample"].'
        raise ValueError(msg)

# 사용 예시
if __name__ == "__main__":
    # GeneralPrompt 사용
    general_extraction_model = GeneralPrompt.get_extraction_model()
    general_prompt = GeneralPrompt.format_user_prompt("sample text", "sample problem")
    print(f"General System: {GeneralPrompt.SYSTEM_INSTRUCTION}")
    
    # SamplePrompt 사용
    sample_extraction_model = SamplePrompt.get_extraction_model()
    sample_prompt = SamplePrompt.format_user_prompt("sample text", "sample problem")
    print(f"Sample System: {SamplePrompt.SYSTEM_INSTRUCTION}")
