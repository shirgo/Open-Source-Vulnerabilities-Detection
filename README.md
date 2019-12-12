# open-source-vulnerabilities-detection

## The Goal
Identify new open source vulnerabilities by checking github activity.

The code provides ranking for a set of github events by their chance to be a real vulnerability.

## The data:
A TSV (Tab separated file) called “Tagged Data.csv” having 9266 rows, with the following fields:
- url (string) - the link to the event in github or in other sources.
- lang (string) - the programming language of this sample (sparse).
- eventType (string) - as described here , with some missing values.
- repo (string - the name of the repository in github.
- title (string) - title of the sample.
- description (string) - description of the sample.
- vuln (boolean) - label indicating if it is a real vuln or not. This was labeled by domain experts, so it is assumed that the tags are 100% accurate.

## The code:
The code contains feature engineering and model training using oversampling.

The code generates and visualizes metrics for model evaluation over train and test data and orders the test data by a score, where higher scores are expected to be more likely of being a vulnerability.

An additional analysis of the score and the important features is done in the code.

