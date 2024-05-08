from src.utils.extract_json import extract_json_only


def test_extract_json_inside_json():
    test_string = """
Here is the extracted information in JSON format:

```
{
    "Skills": [
        {
            "name": "Bid management",
            "description": ""
        },
        {
            "name": "Sales support",
            "description": ""
        },
        {
            "name": "Requirement Analysis",
            "description": "(Less than 1 year)"
        },
        {
            "name": "Test Planning and Test execution",
            "description": ""
        }
    ],
    "Education": [
        {
            "year": 2015,
            "institute": "Great Lakes Institute of Management"
        }
    ],
    "Work Experience": [
        {
            "company": "Infosys Limited",
            "designation": "Senior Associate Consultant"
        }
    ],
    "Location": {
        "city": "",
        "country": "Hyderabad"
    },
    "Contact Information": {
        "email": "indeed.com/r/Abra-Cadabra/contact"
    },
    "Name": "Abra Cadabra"
}
```

Note that the `Skills` section has multiple entries, each with a name and an optional description. The `Education` section only has one entry, which includes the graduation year and college name. The `Work Experience` section also only has one entry, which includes the company name and designation. The `Location` section includes the city and country, but they are not specified in the original text so I left them blank. The `Contact Information` section includes an email address.
"""
    expected = [
        '[\n        {\n            "name": "Bid management",\n            "description": ""\n        },\n        {\n            "name": "Sales support",\n            "description": ""\n        },\n        {\n            "name": "Requirement Analysis",\n            "description": "(Less than 1 year)"\n        },\n        {\n            "name": "Test Planning and Test execution",\n            "description": ""\n        }\n    ]',
        '[\n        {\n            "year": 2015,\n            "institute": "Great Lakes Institute of Management"\n        }\n    ]',
        '[\n        {\n            "company": "Infosys Limited",\n            "designation": "Senior Associate Consultant"\n        }\n    ]',
    ]

    result = extract_json_only(test_string)

    assert result == expected


def test_extract_json_null():
    test_string = """
Here are the extracted job positions along with their attributes:

```
[
    {
        "position_title": "Senior Associate Consultant",
        "company": "Infosys Limited",
        "start_year": null,
        "end_year": null
    }
]
```

Note that there is only one job position mentioned in the resume, and it does not have a specific start or end year. The other sections (such as "Skills", "Graduation Year", etc.) do not contain job positions, so they are not included in the output.
    """

    expected = [
        '[\n    {\n        "position_title": "Senior Associate Consultant",\n        "company": "Infosys Limited",\n        "start_year": null,\n        "end_year": null\n    }\n]'
    ]

    result = extract_json_only(test_string)

    assert result == expected


def test_extract_json_empty_string():
    test_string = """
Here are the extracted job positions along with their attributes:

```
[
    {
        "position_title": "",
        "company": "Infosys Limited",
        "start_year": null,
        "end_year": null
    }
]
```

Note that there is only one job position mentioned in the resume, and it does not have a specific start or end year. The other sections (such as "Skills", "Graduation Year", etc.) do not contain job positions, so they are not included in the output.
    """

    expected = [
        '[\n    {\n        "position_title": "",\n        "company": "Infosys Limited",\n        "start_year": null,\n        "end_year": null\n    }\n]'
    ]

    result = extract_json_only(test_string)

    assert result == expected


def test_extract_json_single_attribute_single_item():
    test_string = """
raw output
Based on the extracted annotation from the resume, I can extract the email address as follows:

```
[
    {"value": "indeed.com/r/Avin-Sharma/3ad8a8b57a172613"}
]
```

Note that this is an unusual format for an email address, and it appears to be a link to a LinkedIn profile rather than a traditional email address. If you're looking for a traditional email address, I wouldn't be able to extract one from this resume.
clean output
    """

    expected = ["""[\n    {"value": "indeed.com/r/Avin-Sharma/3ad8a8b57a172613"}\n]"""]

    result = extract_json_only(test_string)

    assert result == expected


def test_extract_json_single_attributes_multiple_items():
    test_string = """
Based on the extracted annotations from the resume, here are the skills mentioned:

```
[
    {"name": "Bid management"},
    {"name": "Sales support"},
    {"name": "Requirement Analysis"},
    {"name": "Test Planning"},
    {"name": "Test execution"}
]
```
"""

    expected = [
        """[\n    {"name": "Bid management"},\n    {"name": "Sales support"},\n    {"name": "Requirement Analysis"},\n    {"name": "Test Planning"},\n    {"name": "Test execution"}\n]"""
    ]

    result = extract_json_only(test_string)

    assert result == expected
