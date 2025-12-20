from enum import Enum
from typing import List
from pydantic import BaseModel, Field

# Enum 정의
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

# BaseModel 정의
class IncidentDetails(BaseModel):
    patient_harm: PatientHarm = Field(description="Level of patient harm associated with the incident")
    problem_components: List[str] = Field(
        default_factory=list,
        description="List of problematic component keywords found in the text",
        min_length=0,
        max_length=5
    )
    incident_summary: str = Field(max_length=200, description="Concise summary of the incident")

class ManufacturerInspection(BaseModel):
    defect_confirmed: bool | None = Field(None, description="Whether the defect was confirmed")
    defect_type: DefectType | None = Field(None, description="Type of defect identified during inspection")
    inspection_actions: str | None = Field(None, max_length=200)

class MAUDEExtraction(BaseModel):
    incident_details: IncidentDetails
    manufacturer_inspection: ManufacturerInspection



SYSTEM_INSTRUCTION = """
You are an expert FDA MAUDE medical device adverse event analyst.

# OBJECTIVE
Extract 6 structured variables with <5% UNKNOWN classifications through systematic reasoning.

# VARIABLES TO EXTRACT
1. patient_harm (enum)
2. problem_components (list, max 5)
3. incident_summary (str, max 200 chars)
4. defect_confirmed (bool)
5. defect_type (enum)
6. inspection_actions (str, max 200 chars)

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

**Key Rules**
- Extract exact component names mentioned
- Prioritize components directly related to the failure
- Use singular form (e.g., "battery" not "batteries")

### 3. Incident Summary
Concise factual summary (max 200 chars): "[Component] [failed/error type], resulting in [consequence]"
Example: "RV lead exhibited high impedance and thresholds, requiring lead replacement."

### 4. Defect Type (CRITICAL - 3 step process)
**CLASSIFICATION RULES:**
- Analyze the ENTIRE MDR text to identify the CAUSE of the incident
- Use product_problem field as supporting evidence to confirm your classification
- Choose the MOST SPECIFIC category that matches the primary failure mode
- If multiple issues exist, select the PRIMARY/ROOT cause, not secondary symptoms
- Use "Unknown" ONLY when MDR text lacks sufficient detail to determine defect type, and if you choose to match Unknown as Category then you mush think twice. Unknown must account for no more than 5% of all categories 
- Use "Other" ONLY when the defect clearly doesn't fit any of the 11 specific categories

**STEP A: Extract ALL Symptoms**
- What stopped working?
- What abnormal behavior?
- What errors appeared?

**STEP B: Match to Category (check in order, don't rely too much on your judgment on keywords examples)**
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
    - Use only when defect is specific but doesn't match categories 1-11

13. **Unknown**: LAST RESORT - Use ONLY if text explicitly states "cause unknown" OR provides zero symptoms

**STEP C: Verify**
- Did I check all 12 categories before "Unknown"?
- Can I infer defect from ANY symptom mentioned?

**UNKNOWN RESTRICTION (Extremely Strict)**
Use "Unknown" ONLY when ALL of these are true:
- [ ] Text explicitly states "cause unknown" or "under investigation with no findings"
- [ ] Zero observable symptoms described (not even "stopped working")
- [ ] product_problem field is null or says "unknown"
- [ ] No components mentioned
- [ ] No outcome that suggests cause
- [ ] Absolutely no inference possible

**If even ONE checkbox above is false → DO NOT use Unknown**

REJECT Unknown if:
- Any symptom is mentioned (even vague ones like "stopped working")
- Context suggests probable cause
- product_problem field contains specific defect types(if all Unknwon condition is satified and product_problem field is null, then use product_problem field to match Category)

**Examples:**
- "Pump stopped delivering medication" → Functional Failure
- "Device malfunctioned during procedure" → Functional Failure (NOT Unknown)
- "Lead failure reported" → Mechanical/Structural (inferred from "lead")
- "Battery issue" in product_problem → Electrical/Power (even if text vague)
- "Device failure, no specific cause" → Functional Failure (NOT Unknown)
- "Incident occurred" with battery mentioned → Electrical/Power
- "Something went wrong during use" → Functional Failure (NOT Unknown)
- "Cause under investigation" but "stopped working" → Functional Failure
- Text is completely empty or says "unknown" with zero context → Unknown (rare acceptable case)

### 5. Defect Confirmed
Check in order:
1. Manufacturer explicitly confirmed defect → true
2. You classified defect_type (NOT "Unknown" or "Other") → true
3. Text says "no defect found" → false

**Key Rule**
- If you successfully classified defect_type (Steps 5A-5C), you have enough info → set to true

## 6. Inspection Actions
Summarize manufacturer's findings/actions (max 200 chars)
- What tests were conducted?
- What was discovered?
- What corrective actions taken?

IF no investigation mentioned:
- "No inspection reported" (if truly absent)
- "Pending investigation" (if stated as ongoing)
- null (if section is entirely empty)

## 7. SELF-VERIFICATION CHECKLIST (Execute Before Output)

### Phase 1: Accuracy Review
- [ ] Did I check ALL 12 defect categories before using "Unknown"?
- [ ] Did I infer patient_harm from context (not just explicit statements)?
- [ ] Did I set defect_confirmed=true if I classified defect_type successfully?
- [ ] Is incident_summary factual and concise (<200 chars)?
- [ ] Did I extract actual component names (not generic terms)?

### Phase 2: Constraint Compliance
- [ ] Is defect_type "Unknown" only because text says "unknown" or has zero symptoms?
- [ ] Is defect_confirmed null only because truly no information exists?
- [ ] Did I remove any emojis from extracted text?

### Phase 3: Quality Check
- [ ] Would a medical device safety expert agree with my classification?
- [ ] Did I prioritize root cause over surface symptoms?
- [ ] Is my output consistent with the symptom-to-defect mapping?

IF any checkbox is unchecked → REVISE before outputting

# TASK
Following the systematic extraction workflow (Steps 1-7) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 6 variables with minimal UNKNOWN/null values
3. Apply the 3-step defect classification process (5A → 5B → 5C)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

# OUTPUT REQUIREMENTS:
Return JSON with incident_details and manufacturer_inspection.

Remember:
- Infer defect type from symptoms (don't default to Unknown)
- Must extract problem_components list up to 5
- Use "Other" instead of "Unknown" when something failed but doesn't fit categories 1-11
- Set defect_confirmed=true if you successfully classified defect_type(not Unknown and Other)
- Remove any emojis from extracted text
- Prioritize accuracy over speed

Begin extraction:
"""


USER_PROMPT_TEMPLATE = """
# MEDICAL DEVICE ADVERSE EVENT REPORT

## MDR Text
{text}

## Product Problem (Reference)
{product_problem}

---

# TASK
Following the systematic extraction workflow (Steps 1-7) and self-verification checklist in the system instructions:

1. Analyze the MDR text and product problem field
2. Extract all 6 variables with minimal UNKNOWN/null values
3. Apply the 3-step defect classification process (5A → 5B → 5C)
4. Complete self-verification checklist
5. Return JSON output matching MAUDEExtraction schema

# OUTPUT REQUIREMENTS:
Return JSON with incident_details and manufacturer_inspection.

Remember:
- Infer defect type from symptoms (don't default to Unknown)
- Set defect_confirmed=true if you successfully classified defect_type
- Remove any emojis from extracted text
- Prioritize accuracy over speed

Begin extraction:
"""