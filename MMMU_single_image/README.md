---
language:
- en
license: apache-2.0
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- visual-question-answering
- multiple-choice
pretty_name: mmmu
dataset_info:
- config_name: Accounting
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 262599.0
    num_examples: 5
  - name: validation
    num_bytes: 1598285.0
    num_examples: 30
  - name: test
    num_bytes: 22149332
    num_examples: 380
  download_size: 59029754
  dataset_size: 24010216.0
- config_name: Agriculture
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 22082656.0
    num_examples: 5
  - name: validation
    num_bytes: 119217558.0
    num_examples: 30
  - name: test
    num_bytes: 993778353
    num_examples: 287
  download_size: 2151854185
  dataset_size: 1135078567.0
- config_name: Architecture_and_Engineering
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 137750.0
    num_examples: 5
  - name: validation
    num_bytes: 721378.0
    num_examples: 30
  - name: test
    num_bytes: 16054037
    num_examples: 551
  download_size: 64641185
  dataset_size: 16913165.0
- config_name: Art
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 6241184.0
    num_examples: 5
  - name: validation
    num_bytes: 29934534.0
    num_examples: 30
  - name: test
    num_bytes: 237879561
    num_examples: 231
  download_size: 823627085
  dataset_size: 274055279.0
- config_name: Art_Theory
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 7435106.0
    num_examples: 5
  - name: validation
    num_bytes: 33481558.0
    num_examples: 30
  - name: test
    num_bytes: 553187064
    num_examples: 429
  download_size: 1483719297
  dataset_size: 594103728.0
- config_name: Basic_Medical_Science
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 814310.0
    num_examples: 5
  - name: validation
    num_bytes: 4125930.0
    num_examples: 30
  - name: test
    num_bytes: 48133203
    num_examples: 326
  download_size: 132732889
  dataset_size: 53073443.0
- config_name: Biology
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 574342.0
    num_examples: 5
  - name: validation
    num_bytes: 8491863.0
    num_examples: 30
  - name: test
    num_bytes: 132969700
    num_examples: 345
  download_size: 540589709
  dataset_size: 142035905.0
- config_name: Chemistry
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 262397.0
    num_examples: 5
  - name: validation
    num_bytes: 1518573.0
    num_examples: 30
  - name: test
    num_bytes: 37266424
    num_examples: 603
  download_size: 145231148
  dataset_size: 39047394.0
- config_name: Clinical_Medicine
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 1467945.0
    num_examples: 5
  - name: validation
    num_bytes: 10882484.0
    num_examples: 30
  - name: test
    num_bytes: 98213499
    num_examples: 325
  download_size: 258723529
  dataset_size: 110563928.0
- config_name: Computer_Science
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 440523.0
    num_examples: 5
  - name: validation
    num_bytes: 2072018.0
    num_examples: 30
  - name: test
    num_bytes: 32056550
    num_examples: 371
  download_size: 86542553
  dataset_size: 34569091.0
- config_name: Design
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 2259873.0
    num_examples: 5
  - name: validation
    num_bytes: 17923120.0
    num_examples: 30
  - name: test
    num_bytes: 77680347
    num_examples: 169
  download_size: 220116911
  dataset_size: 97863340.0
- config_name: Diagnostics_and_Laboratory_Medicine
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 2056117.0
    num_examples: 5
  - name: validation
    num_bytes: 37106233.0
    num_examples: 30
  - name: test
    num_bytes: 157049246
    num_examples: 162
  download_size: 760854338
  dataset_size: 196211596.0
- config_name: Economics
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 171434.0
    num_examples: 5
  - name: validation
    num_bytes: 1487048.0
    num_examples: 30
  - name: test
    num_bytes: 11854869
    num_examples: 267
  download_size: 31967167
  dataset_size: 13513351.0
- config_name: Electronics
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 123632.0
    num_examples: 5
  - name: validation
    num_bytes: 641377.0
    num_examples: 30
  - name: test
    num_bytes: 5755772
    num_examples: 256
  download_size: 17118620
  dataset_size: 6520781.0
- config_name: Energy_and_Power
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 105006.0
    num_examples: 5
  - name: validation
    num_bytes: 1641935.0
    num_examples: 30
  - name: test
    num_bytes: 14749680
    num_examples: 432
  download_size: 49859270
  dataset_size: 16496621.0
- config_name: Finance
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 296124.0
    num_examples: 5
  - name: validation
    num_bytes: 1071060.0
    num_examples: 30
  - name: test
    num_bytes: 12069966
    num_examples: 355
  download_size: 41140709
  dataset_size: 13437150.0
- config_name: Geography
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 1494060.0
    num_examples: 5
  - name: validation
    num_bytes: 6671316.0
    num_examples: 30
  - name: test
    num_bytes: 137217772
    num_examples: 565
  download_size: 511199153
  dataset_size: 145383148.0
- config_name: History
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 1444231.0
    num_examples: 5
  - name: validation
    num_bytes: 8819857.0
    num_examples: 30
  - name: test
    num_bytes: 115330404
    num_examples: 278
  download_size: 347749137
  dataset_size: 125594492.0
- config_name: Literature
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 2451201.0
    num_examples: 5
  - name: validation
    num_bytes: 14241046.0
    num_examples: 30
  - name: test
    num_bytes: 50301429
    num_examples: 112
  download_size: 180591942
  dataset_size: 66993676.0
- config_name: Manage
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 449514.0
    num_examples: 5
  - name: validation
    num_bytes: 3277436.0
    num_examples: 30
  - name: test
    num_bytes: 29972520
    num_examples: 245
  download_size: 80748903
  dataset_size: 33699470.0
- config_name: Marketing
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 116960.0
    num_examples: 5
  - name: validation
    num_bytes: 1472981.0
    num_examples: 30
  - name: test
    num_bytes: 7732798
    num_examples: 181
  download_size: 20184823
  dataset_size: 9322739.0
- config_name: Materials
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 239632.0
    num_examples: 5
  - name: validation
    num_bytes: 2305223.0
    num_examples: 30
  - name: test
    num_bytes: 25311708
    num_examples: 458
  download_size: 130939055
  dataset_size: 27856563.0
- config_name: Math
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 175839.0
    num_examples: 5
  - name: validation
    num_bytes: 1444496.0
    num_examples: 30
  - name: test
    num_bytes: 27806055
    num_examples: 505
  download_size: 201728306
  dataset_size: 29426390.0
- config_name: Mechanical_Engineering
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 152542.0
    num_examples: 5
  - name: validation
    num_bytes: 874988.0
    num_examples: 30
  - name: test
    num_bytes: 15094272
    num_examples: 429
  download_size: 45403173
  dataset_size: 16121802.0
- config_name: Music
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 1417615.0
    num_examples: 5
  - name: validation
    num_bytes: 9359372.0
    num_examples: 30
  - name: test
    num_bytes: 134101164
    num_examples: 334
  download_size: 307658022
  dataset_size: 144878151.0
- config_name: Pharmacy
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 207924.0
    num_examples: 5
  - name: validation
    num_bytes: 1656342.0
    num_examples: 30
  - name: test
    num_bytes: 31971026
    num_examples: 430
  download_size: 93950435
  dataset_size: 33835292.0
- config_name: Physics
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 233734.0
    num_examples: 5
  - name: validation
    num_bytes: 1114130.0
    num_examples: 30
  - name: test
    num_bytes: 15929535
    num_examples: 408
  download_size: 51033214
  dataset_size: 17277399.0
- config_name: Psychology
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 600864.0
    num_examples: 5
  - name: validation
    num_bytes: 4403886.0
    num_examples: 30
  - name: test
    num_bytes: 53829989
    num_examples: 305
  download_size: 154215036
  dataset_size: 58834739.0
- config_name: Public_Health
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 234781.0
    num_examples: 5
  - name: validation
    num_bytes: 1508761.0
    num_examples: 30
  - name: test
    num_bytes: 32149632
    num_examples: 509
  download_size: 79953306
  dataset_size: 33893174.0
- config_name: Sociology
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: options
    dtype: string
  - name: explanation
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: image_5
    dtype: image
  - name: image_6
    dtype: image
  - name: image_7
    dtype: image
  - name: img_type
    dtype: string
  - name: answer
    dtype: string
  - name: topic_difficulty
    dtype: string
  - name: question_type
    dtype: string
  - name: subfield
    dtype: string
  splits:
  - name: dev
    num_bytes: 3769220.0
    num_examples: 5
  - name: validation
    num_bytes: 18455336.0
    num_examples: 30
  - name: test
    num_bytes: 144301819
    num_examples: 252
  download_size: 454539538
  dataset_size: 166526375.0
configs:
- config_name: Accounting
  data_files:
  - split: dev
    path: Accounting/dev-*
  - split: validation
    path: Accounting/validation-*
  - split: test
    path: Accounting/test-*
- config_name: Agriculture
  data_files:
  - split: dev
    path: Agriculture/dev-*
  - split: validation
    path: Agriculture/validation-*
  - split: test
    path: Agriculture/test-*
- config_name: Architecture_and_Engineering
  data_files:
  - split: dev
    path: Architecture_and_Engineering/dev-*
  - split: validation
    path: Architecture_and_Engineering/validation-*
  - split: test
    path: Architecture_and_Engineering/test-*
- config_name: Art
  data_files:
  - split: dev
    path: Art/dev-*
  - split: validation
    path: Art/validation-*
  - split: test
    path: Art/test-*
- config_name: Art_Theory
  data_files:
  - split: dev
    path: Art_Theory/dev-*
  - split: validation
    path: Art_Theory/validation-*
  - split: test
    path: Art_Theory/test-*
- config_name: Basic_Medical_Science
  data_files:
  - split: dev
    path: Basic_Medical_Science/dev-*
  - split: validation
    path: Basic_Medical_Science/validation-*
  - split: test
    path: Basic_Medical_Science/test-*
- config_name: Biology
  data_files:
  - split: dev
    path: Biology/dev-*
  - split: validation
    path: Biology/validation-*
  - split: test
    path: Biology/test-*
- config_name: Chemistry
  data_files:
  - split: dev
    path: Chemistry/dev-*
  - split: validation
    path: Chemistry/validation-*
  - split: test
    path: Chemistry/test-*
- config_name: Clinical_Medicine
  data_files:
  - split: dev
    path: Clinical_Medicine/dev-*
  - split: validation
    path: Clinical_Medicine/validation-*
  - split: test
    path: Clinical_Medicine/test-*
- config_name: Computer_Science
  data_files:
  - split: dev
    path: Computer_Science/dev-*
  - split: validation
    path: Computer_Science/validation-*
  - split: test
    path: Computer_Science/test-*
- config_name: Design
  data_files:
  - split: dev
    path: Design/dev-*
  - split: validation
    path: Design/validation-*
  - split: test
    path: Design/test-*
- config_name: Diagnostics_and_Laboratory_Medicine
  data_files:
  - split: dev
    path: Diagnostics_and_Laboratory_Medicine/dev-*
  - split: validation
    path: Diagnostics_and_Laboratory_Medicine/validation-*
  - split: test
    path: Diagnostics_and_Laboratory_Medicine/test-*
- config_name: Economics
  data_files:
  - split: dev
    path: Economics/dev-*
  - split: validation
    path: Economics/validation-*
  - split: test
    path: Economics/test-*
- config_name: Electronics
  data_files:
  - split: dev
    path: Electronics/dev-*
  - split: validation
    path: Electronics/validation-*
  - split: test
    path: Electronics/test-*
- config_name: Energy_and_Power
  data_files:
  - split: dev
    path: Energy_and_Power/dev-*
  - split: validation
    path: Energy_and_Power/validation-*
  - split: test
    path: Energy_and_Power/test-*
- config_name: Finance
  data_files:
  - split: dev
    path: Finance/dev-*
  - split: validation
    path: Finance/validation-*
  - split: test
    path: Finance/test-*
- config_name: Geography
  data_files:
  - split: dev
    path: Geography/dev-*
  - split: validation
    path: Geography/validation-*
  - split: test
    path: Geography/test-*
- config_name: History
  data_files:
  - split: dev
    path: History/dev-*
  - split: validation
    path: History/validation-*
  - split: test
    path: History/test-*
- config_name: Literature
  data_files:
  - split: dev
    path: Literature/dev-*
  - split: validation
    path: Literature/validation-*
  - split: test
    path: Literature/test-*
- config_name: Manage
  data_files:
  - split: dev
    path: Manage/dev-*
  - split: validation
    path: Manage/validation-*
  - split: test
    path: Manage/test-*
- config_name: Marketing
  data_files:
  - split: dev
    path: Marketing/dev-*
  - split: validation
    path: Marketing/validation-*
  - split: test
    path: Marketing/test-*
- config_name: Materials
  data_files:
  - split: dev
    path: Materials/dev-*
  - split: validation
    path: Materials/validation-*
  - split: test
    path: Materials/test-*
- config_name: Math
  data_files:
  - split: dev
    path: Math/dev-*
  - split: validation
    path: Math/validation-*
  - split: test
    path: Math/test-*
- config_name: Mechanical_Engineering
  data_files:
  - split: dev
    path: Mechanical_Engineering/dev-*
  - split: validation
    path: Mechanical_Engineering/validation-*
  - split: test
    path: Mechanical_Engineering/test-*
- config_name: Music
  data_files:
  - split: dev
    path: Music/dev-*
  - split: validation
    path: Music/validation-*
  - split: test
    path: Music/test-*
- config_name: Pharmacy
  data_files:
  - split: dev
    path: Pharmacy/dev-*
  - split: validation
    path: Pharmacy/validation-*
  - split: test
    path: Pharmacy/test-*
- config_name: Physics
  data_files:
  - split: dev
    path: Physics/dev-*
  - split: validation
    path: Physics/validation-*
  - split: test
    path: Physics/test-*
- config_name: Psychology
  data_files:
  - split: dev
    path: Psychology/dev-*
  - split: validation
    path: Psychology/validation-*
  - split: test
    path: Psychology/test-*
- config_name: Public_Health
  data_files:
  - split: dev
    path: Public_Health/dev-*
  - split: validation
    path: Public_Health/validation-*
  - split: test
    path: Public_Health/test-*
- config_name: Sociology
  data_files:
  - split: dev
    path: Sociology/dev-*
  - split: validation
    path: Sociology/validation-*
  - split: test
    path: Sociology/test-*
tags:
- biology
- medical
- finance
- chemistry
- music
- art
- art_theory
- design
- music
- business
- accounting
- economics
- finance
- manage
- marketing
- health
- medicine
- basic_medical_science
- clinical
- pharmacy
- public_health
- humanities
- social_science
- history
- literature
- sociology
- psychology
- science
- biology
- chemistry
- geography
- math
- physics
- engineering
- agriculture
- architecture
- computer_science
- electronics
- energy_and_power
- materials
- mechanical_engineering
---


# MMMU (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI)

[**🌐 Homepage**](https://mmmu-benchmark.github.io/) | [**🏆 Leaderboard**](https://mmmu-benchmark.github.io/#leaderboard) | [**🤗 Dataset**](https://huggingface.co/datasets/MMMU/MMMU/) | [**🤗 Paper**](https://huggingface.co/papers/2311.16502) | [**📖 arXiv**](https://arxiv.org/abs/2311.16502) | [**GitHub**](https://github.com/MMMU-Benchmark/MMMU)

## 🔔News

- **‼️[2026-02-12] We have released the answers for the test set! You can now evaluate your models on the test set locally! 🎉**
- **🛠️[2024-05-30]: Fixed duplicate option issues in Materials dataset items (validation_Materials_25; test_Materials_17, 242) and content error in validation_Materials_25.**
- **🛠️[2024-04-30]: Fixed missing "-" or "^" signs in Math dataset items (dev_Math_2, validation_Math_11, 12, 16; test_Math_8, 23, 43, 113, 164, 223, 236, 287, 329, 402, 498) and corrected option errors in validation_Math_2. If you encounter any issues with the dataset, please contact us promptly!**
- **🚀[2024-01-31]: We added Human Expert performance on the [Leaderboard](https://mmmu-benchmark.github.io/#leaderboard)!🌟**
- **🔥[2023-12-04]: ~~Our evaluation server for test set is now availble on [EvalAI](https://eval.ai/web/challenges/challenge-page/2179/overview).~~ We welcome all submissions and look forward to your participation! 😆**

## Dataset Details

### Dataset Description

We introduce MMMU: a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes **11.5K meticulously collected multimodal questions** from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span **30 subjects** and **183 subfields**, comprising **30 highly heterogeneous image types**, such as charts, diagrams, maps, tables, music sheets, and chemical structures. We believe MMMU will stimulate the community to build next-generation multimodal foundation models towards expert artificial general intelligence (AGI).

🎯 **We have released a full set comprising 150 development samples, 900 validation samples and 10,500 test samples.** 
The development set is used for few-shot/in-context learning, and the validation set is used for debugging models, selecting hyperparameters, or quick evaluations. ~~The answers and explanations for the test set questions are withheld. You can submit your model's predictions for the **test set** on **[EvalAI](https://eval.ai/web/challenges/challenge-page/2179/overview)**.~~

The answers and explanations for the test set samples are now released. You can evaluate your models locally!

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6230d750d93e84e233882dbc/2Ulh9yznm1dvISV4xJ_Ok.png)

### Dataset Creation

MMMU was created to challenge multimodal models with tasks that demand college-level subject knowledge and deliberate reasoning, pushing the boundaries of what these models can achieve in terms of expert-level perception and reasoning.
The data for the MMMU dataset was manually collected by a team of college students from various disciplines, using online sources, textbooks, and lecture materials.

- **Content:** The dataset contains 11.5K college-level problems across six broad disciplines (Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering) and 30 college subjects.
- **Image Types:** The dataset includes 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures, interleaved with text.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6230d750d93e84e233882dbc/Mbf8O5lEH8I8czprch0AG.png)


## 🏆 Mini-Leaderboard
We show a mini-leaderboard here and please find more information in our paper or [**homepage**](https://mmmu-benchmark.github.io/).

| Model                          | Val (900) | Test (10.5K) |
|--------------------------------|:---------:|:------------:|
| Expert (Best)                  |   88.6    |      -       |
| Expert (Medium)                |   82.6    |      -       |
| Expert (Worst)                 |   76.2    |      -       |
| GPT-4o*                        | **69.1**  |      -       |
| Gemini 1.5 Pro*                |   62.2    |      -       |
| InternVL2-Pro*                 |   62.0    |   **55.7**   |
| Gemini 1.0 Ultra*              |   59.4    |      -       |
| Claude 3 Opus*                 |   59.4    |      -       |
| GPT-4V(ision) (Playground)     |   56.8    |   **55.7**   |
| Reka Core*                     |   56.3    |      -       |
| Gemini 1.5 Flash*              |   56.1    |      -       |
| SenseChat-Vision-0423-Preview* |   54.6    |     50.3     |
| Reka Flash*                    |   53.3    |      -       |
| Claude 3 Sonnet*               |   53.1    |      -       |
| HPT Pro*                       |   52.0    |      -       |
| VILA1.5*                       |   51.9    |     46.9     |
| Qwen-VL-MAX*                   |   51.4    |     46.8     |
| InternVL-Chat-V1.2*            |   51.6    |     46.2     |
| Skywork-VL*                    |   51.4    |     46.2     |
| LLaVA-1.6-34B*                 |   51.1    |     44.7     |
| Claude 3 Haiku*                |   50.2    |      -       |
| Adept Fuyu-Heavy*              |   48.3    |      -       |
| Gemini 1.0 Pro*                |   47.9    |      -       |
| Marco-VL-Plus*                 |   46.2    |     44.3     |
| Yi-VL-34B*                     |   45.9    |     41.6     |
| Qwen-VL-PLUS*                  |   45.2    |     40.8     |
| HPT Air*                       |   44.0    |      -       |
| Reka Edge*                     |   42.8    |      -       |
| Marco-VL*                      |   41.2    |     40.4     |
| OmniLMM-12B*                   |   41.1    |     40.4     |
| Bunny-8B*                      |   43.3    |     39.0     |
| Bunny-4B*                      |   41.4    |     38.4     |
| Weitu-VL-1.0-15B*              |     -     |     38.4     |
| InternLM-XComposer2-VL*        |   43.0    |     38.2     |
| Yi-VL-6B*                      |   39.1    |     37.8     |
| InfiMM-Zephyr-7B*              |   39.4    |     35.5     |
| InternVL-Chat-V1.1*            |   39.1    |     35.3     |
| Math-LLaVA-13B*                |   38.3    |     34.6     |
| SVIT*                          |   38.0    |     34.1     |
| MiniCPM-V*                     |   37.2    |     34.1     |
| MiniCPM-V-2*                   |   37.1    |      -       |
| Emu2-Chat*                     |   36.3    |     34.1     |
| BLIP-2 FLAN-T5-XXL             |   35.4    |     34.0     |
| InstructBLIP-T5-XXL            |   35.7    |     33.8     |
| LLaVA-1.5-13B                  |   36.4    |     33.6     |
| Bunny-3B*                      |   38.2    |     33.0     |
| Qwen-VL-7B-Chat                |   35.9    |     32.9     |
| SPHINX*                        |   32.9    |     32.9     |
| mPLUG-OWL2*                    |   32.7    |     32.1     |
| BLIP-2 FLAN-T5-XL              |   34.4    |     31.0     |
| InstructBLIP-T5-XL             |   32.9    |     30.6     |
| Gemini Nano2*                  |   32.6    |      -       |
| CogVLM                         |   32.1    |     30.1     |
| Otter                          |   32.2    |     29.1     |
| LLaMA-Adapter2-7B              |   29.8    |     27.7     |
| MiniGPT4-Vicuna-13B            |   26.8    |     27.6     |
| Adept Fuyu-8B                  |   27.9    |     27.4     |
| Kosmos2                        |   24.4    |     26.6     |
| OpenFlamingo2-9B               |   28.7    |     26.3     |
| Frequent Choice                |   22.1    |     23.9     |
| Random Choice                  |   26.8    |     25.8     |

*: results provided by the authors.


## Limitations
Despite its comprehensive nature, MMMU, like any benchmark, is not without limitations. The manual curation process, albeit thorough, may carry biases. 
And the focus on college-level subjects might not fully be a sufficient test for Expert AGI. 
However, we believe it should be necessary for an Expert AGI to achieve strong performance on MMMU to demonstrate their broad and deep subject knowledge as well as expert-level understanding and reasoning capabilities. 
In future work, we plan to incorporate human evaluations into MMMU. This will provide a more grounded comparison between model capabilities and expert performance, shedding light on the proximity of current AI systems to achieving Expert AGI. 

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to notify us. Upon verification, such samples will be promptly removed.

## Contact
- Xiang Yue: xiangyue.work@gmail.com
- Yu Su: su.809@osu.edu
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@inproceedings{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun and Yu Su and Wenhu Chen},
  booktitle={Proceedings of CVPR},
  year={2024},
}
```