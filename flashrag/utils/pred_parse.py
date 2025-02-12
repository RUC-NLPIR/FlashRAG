import re

def selfask_pred_parse(dataset):
    """Parsing the prediction results of self-ask format."""
    FINAL_ANSWER_PREFIX = "So the final answer is: "

    for item in dataset:
        pred = item.pred
        lines = pred.split("\n")
        answer = ""
        for line in lines:
            if FINAL_ANSWER_PREFIX in line:
                answer = line.split(FINAL_ANSWER_PREFIX)[1].strip()
                break
        item.update_output('raw_pred', pred)
        item.update_output('pred', answer)

    return dataset


def ircot_pred_parse(dataset):
    FINAL_ANSWER_PREFIX = "So the answer is:"
    for item in dataset:
        pred = item.pred
        if FINAL_ANSWER_PREFIX in pred:
            answer = pred.split(FINAL_ANSWER_PREFIX)[1].strip()
        else:
            answer = pred
        item.update_output('raw_pred', pred)
        item.update_output('pred', answer)
    return dataset


def basic_pred_parse(dataset):
    for item in dataset:
        pred = item.pred
        item.update_output('raw_pred', pred)
        item.update_output('pred', pred.split("\n")[0].strip())
    return dataset



def gaokaomm_pred_parse(dataset):
    """
    Extract choice answer from model output.

    Format of model_output that is expected:
    'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
    """

    for item in dataset:
        model_output = item.pred
        question_type = item.question_type

        if question_type == 'single_choice':
            model_answer = ""
            temp = re.findall(r'[A-D]', model_output[::-1])
            if len(temp) != 0:
                model_answer = temp[0]

        elif question_type == 'multiple_choice':
            model_answer = []
            answer = ''
            content = re.sub(r'\s+', '', model_output)
            answer_index = content.find('【答案】')
            if answer_index > 0:
                temp = content[answer_index:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            else:
                temp = content[-10:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            if len(answer) != 0:
                model_answer.append(answer)
        model_answer = "".join(model_answer)

        item.update_output('raw_pred', model_output)
        item.update_output('pred', model_answer)

    return dataset