def selfask_pred_parse(pred):
    """Parsing the prediction results of self-ask format."""
    FINAL_ANSWER_PREFIX = "So the final answer is: "

    lines = pred.split("\n")
    answer = ""
    for line in lines:
        if FINAL_ANSWER_PREFIX in line:
            answer = line.split(FINAL_ANSWER_PREFIX)[1].strip()
            break

    return answer


def ircot_pred_parse(pred):
    FINAL_ANSWER_PREFIX = "So the answer is:"
    if FINAL_ANSWER_PREFIX in pred:
        answer = pred.split(FINAL_ANSWER_PREFIX)[1].strip()
    else:
        answer = pred
    return answer


def basic_pred_parse(pred):
    return pred.split("\n")[0].strip()
