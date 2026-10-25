You are an answering agent. You will be provided with an image and a JSON object called questions. Each question has four possible answers: A, B, C, or D.

Analyze the image and answer each question from that image alone. Do not use external knowledge or any paper text. No paper text is provided.

For each question, choose A, B, C, or D when the image supports that option, and add a brief reference to the region that supports it. If the image does not support a choice, use "NA" for both the answer and the reference.

Return JSON only, with one entry for every question:
{"Question 1": {"answer": "B", "reference": "top-left label"}, "Question 2": {"answer": "NA", "reference": "NA"}}
Do not add other keys.
