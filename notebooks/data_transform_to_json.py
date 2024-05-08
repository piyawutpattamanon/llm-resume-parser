# %%
import json
import datetime

resumes = []
with open('Entity Recognition in Resumes.jsonl') as f:
    for line in f:
        resume = json.loads(line)
        resumes.append(resume)

# %%


def transform_resume_to_json(resume_index):
    global resumes

    print('.'  * 80)
    print(f'{datetime.datetime.now()} Processing Resume {resume_index}')
    print('.'  * 80)

    






    resume = resumes[resume_index]

    import os
    file_path = f"data/evaluation/json/resume_{resume_index}.json"
    if os.path.exists(file_path):
        print('the file already exists')
        return
    
    import subprocess
    output = subprocess.run(["say", f"Processing Resume {resume_index}"], capture_output=True, text=True)

    # %%
    prompt = ''

    prompt += '''
    extract jobs positions along with attributes of the position: title, company, start_year, end_year
    job positions only, not projects.

    example output
    [
        {"position_title": "aaa", "company":"bbb", "start_year": null, "end_year": 2023},
        {"position_title": "aaa", "company":"bbb", "start_year": 2021, "end_year": null}
    ]


    '''

    prompt = """
    below is extract annotation from a resume      
    ====

    """

    for item in resume['annotation']:
        prompt += f"=== {item['label']=} ===\n"
        for point in item['points']:
            prompt += f"{point['text']}\n"
        prompt += "\n"

    print(prompt)

    # %%
    # from src.llm.lmstudio import do_llm_completion

    # messages = [
    #     {
    #         "role": "system",
    #         "content": """
    # extract jobs positions along with attributes of the position: title, company, start_year, end_year
    # job positions only, not projects.

    # example output
    # [
    #     {"position_title": "aaa", "company":"bbb", "start_year": null, "end_year": 2023},
    #     {"position_title": "aaa", "company":"bbb", "start_year": 2021, "end_year": null}
    # ]
    # """
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt,
    #     }
    # ]

    # llm_response = do_llm_completion(messages)

    # print('raw output')
    # print(llm_response)

    # print('clean output')
    # from src.utils.extract_json import extract_json_only
    # response_json = extract_json_only(llm_response)
    # print(json.dumps(json.loads(response_json[0]), indent=4))


    # %%
    from src.llm.lmstudio import do_llm_completion

    messages = [
        {
            "role": "system",
            "content": """
    you are resume parser.
    extract email address of this person from his resume.
    output in json

    example output
    [
        { "value": "xxx@gmail.com" }
    ]

    if there is no e-mail in his resume. output a blank array like this
    [ ]
    """
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    llm_response = do_llm_completion(messages)

    print('raw output')
    print(llm_response)

    print('clean output')
    from src.utils.extract_json import extract_json_only
    response_json = extract_json_only(llm_response)
    

    extracted_emails = []
    if len(response_json) > 0:
        extracted_emails = json.loads(response_json[0])
        print(json.dumps(json.loads(response_json[0]), indent=4))

    # %%
    from src.llm.lmstudio import do_llm_completion

    messages = [
        {
            "role": "system",
            "content": """
    you are resume parser.
    extract phone number of this person from his resume.
    output in json

    example output
    [{ "type": "Telephone", "value": "4204208484" }]

    if there is no phone number in his resume. output a blank array like this
    [ ]
    """
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    llm_response = do_llm_completion(messages)

    print('raw output')
    print(llm_response)

    print('clean output')
    from src.utils.extract_json import extract_json_only
    response_json = extract_json_only(llm_response)
    print(response_json)

    extracted_phones = []
    if len(response_json) > 0:
        extracted_emails = json.loads(response_json[0])

    # %%
    from src.llm.lmstudio import do_llm_completion

    messages = [
        {
            "role": "system",
            "content": """
    you are resume parser.
    extract his job positions and information related to each position.
    output in json

    example output
    [
        {
        "city": "",
        "title": "NCAT ASEC",
        "country": "",
        "employer": "Hickman Property Holdings LLC",
        "end_date": "2023-",
        "start_date": "2022-2",
        "country_code": ""
        }
    ]

    if there is no job position in his resume. output a blank array like this
    [ ]
    """
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    llm_response = do_llm_completion(messages)

    print('raw output')
    print(llm_response)

    print('clean output')
    from src.utils.extract_json import extract_json_only
    response_json = extract_json_only(llm_response)
    print(response_json)

    extracted_jobs = []
    if len(response_json) > 0:
        extracted_jobs = json.loads(response_json[0])
        print(json.dumps(json.loads(response_json[0]), indent=4))

    # %%
    from src.llm.lmstudio import do_llm_completion

    messages = [
        {
            "role": "system",
            "content": """
    you are resume parser.
    extract his skills as specified in his resume. both hard and soft skills.
    output in json

    example output
    [
        {"name": "Community Service"},
        {"name": "Concord"}
    ]

    if there is no skills in his resume, output a blank array like this
    [ ]
    """
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    llm_response = do_llm_completion(messages)

    print('raw output')
    print(llm_response)

    print('clean output')
    from src.utils.extract_json import extract_json_only
    response_json = extract_json_only(llm_response)
    print(response_json)
    

    extracted_skills = []
    if len(response_json) > 0:
        extracted_skills = json.loads(response_json[0])
        print(json.dumps(json.loads(response_json[0]), indent=4))

    # %%
    from src.llm.lmstudio import do_llm_completion

    messages = [
        {
            "role": "system",
            "content": """
    you are resume parser.
    extract his education as specified in his resume.
    output in json

    example output
    [
    {
        "city": "",
        "school": "State University",
        "country": "",
        "end_date": "2024-5",
        "start_date": "-",
        "degree_name": "",
        "description": "",
        "country_code": "NC",
        "degree_major": ""
    },
    {
        "city": "",
        "school": "Central Piedmont Community College",
        "country": "",
        "end_date": "2021-12",
        "start_date": "2020-8",
        "degree_name": "",
        "description": "North Carolina Agricultural and Technical State University, Greensboro, NC\nMay 2024\nGPA: 3.4\nCentral Piedmont Community College, Charlotte NC\nAugust 2020 - December 2021\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023",
        "country_code": "NC",
        "degree_major": ""
    }
    ]

    if there is no education in his resume, output a blank array like this
    [ ]
    """
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    llm_response = do_llm_completion(messages)

    print('raw output')
    print(llm_response)

    print('clean output')
    from src.utils.extract_json import extract_json_only
    response_json = extract_json_only(llm_response)
    print(response_json)
    

    extracted_educations = []
    if len(response_json) > 0:
        extracted_educations = json.loads(response_json[0])
        print(json.dumps(json.loads(response_json[0]), indent=4))


    # %%
    output_data = {
        "personal": {
        "gender": "",
        "full_name": "Hidden",
        "birthplace": "",
        "first_name": "Hidden",
        "family_name": "Hidden",
        "middle_name": "Hidden",
        "nationality": [],
        "picture_url": None,
        "date_of_birth": "",
        "marital_status": "",
        "picture_extension": None
    },
    "contact": {
        "email": [{ "value": "xxx@gmail.com" }],
        "phone": [{ "type": "Telephone", "value": "4204208484" }],
        "address": [],
        "website": []
    },
    "summary": {
        "benefits": "",
        "objective": "",
        "description": "",
        "notice_period": "",
        "current_salary": ""
    },
    "metadata": {
        "job_pk": 463152,
        "remark": "Parsing Complete",
        "status": "succeeded",
        "resume_pk": 15801024,
        "candidate_pk": 21380189,
        "language_code": "en",
        "language_confidence": 0.9999951854043442,
    },
    "education": [
        {
        "city": "",
        "school": "State University",
        "country": "",
        "end_date": "2024-5",
        "start_date": "-",
        "degree_name": "",
        "description": "",
        "country_code": "NC",
        "degree_major": "",
        "custom_sections": []
        },
        {
        "city": "",
        "school": "Central Piedmont Community College",
        "country": "",
        "end_date": "2021-12",
        "start_date": "2020-8",
        "degree_name": "",
        "description": "North Carolina Agricultural and Technical State University, Greensboro, NC\nMay 2024\nGPA: 3.4\nCentral Piedmont Community College, Charlotte NC\nAugust 2020 - December 2021\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023\nGPA: 3.3\nCourse Careers, Remote December 2022 - January 2023",
        "country_code": "NC",
        "degree_major": "",
        "custom_sections": []
        }
    ],
    
    "experience": [
        {
        "city": "",
        "title": "NCAT ASEC",
        "country": "",
        "employer": "Hickman Property Holdings LLC",
        "end_date": "2023-",
        "start_date": "2022-2",
        "description": "● A group of like-minded individuals who also are involved on and off campus community service activities",
        "country_code": "",
        "custom_sections": []
        },
        {
        "city": "",
        "title": "Founder",
        "country": "",
        "employer": "Hickman Property Holdings LLC",
        "end_date": "2023-",
        "start_date": "2022-9",
        "description": "● I have 6 employees that I manage and work within my company.",
        "country_code": "",
        "custom_sections": []
        },
        {
        "city": "",
        "title": "Merchandiser",
        "country": "",
        "employer": "Fedex",
        "end_date": "2022-",
        "start_date": "2021-",
        "description": "● Provided great customer service by being knowledgeable about all the consu(...)",
        "country_code": "NC",
        "custom_sections": []
        },
        {
        "city": "",
        "title": "Retail Sales Specialist",
        "country": "",
        "employer": "Fedex",
        "end_date": "2022-",
        "start_date": "2022-",
        "description": "● Provide the Best Customer Service\nProvide the best sales experience(...)",
        "country_code": "",
        "custom_sections": []
        },
        {
        "city": "",
        "title": "Team Lead",
        "country": "",
        "employer": "Burger King",
        "end_date": "2020-",
        "start_date": "2018-",
        "description": "● Utilize leadership skills to properly train new employees",
        "country_code": "NC",
        "custom_sections": []
        }
    ],
        "languages": [{ "code": "en", "name": "English", "description": "I am fluent - Level C1"}],
    "skills": [
        "Community Service",
        "Concord",
        "FedEx",
        "Hard Labor",
        "Leadership",
        "Textbooks",
        "Toys"
    ],
    "achievements": {},
    "certifications": {},
    "qualifications": {}
    }

    # %%
    output_data['skills'] = [item['name'] for item in extracted_skills]
    output_data['education'] = extracted_educations
    output_data['experience'] = extracted_jobs
    output_data['contact']['email'] = extracted_emails
    output_data['contact']['phone'] = extracted_phones

    # %%
    print(json.dumps(output_data, indent=4))

    # %%
    file_path = f"data/evaluation/json/resume_{resume_index}.json"
    print('write to file:', file_path)
    with open(file_path, 'w') as f:
        f.write(json.dumps(output_data, indent=4))


for resume_index in range(len(resumes)):
    try:
        transform_resume_to_json(resume_index)
    except Exception as e:
        import subprocess
        output = subprocess.run(["say", f"Fail at resume {resume_index}"], capture_output=True, text=True)