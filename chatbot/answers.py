def sugang_basic_answer():
    return "수강신청은 어쩌고"

def descision_sugang(sent):
    if (("신청" in sent)):
        return sugang_basic_answer()
    else:
        return -1

def welfare_basic_answer():
    return "학생복지는 어쩌고 저쩌고"

def descision_welfare(sent):
    if (("복지" in sent)):
        return welfare_basic_answer()
    else:
        return -1