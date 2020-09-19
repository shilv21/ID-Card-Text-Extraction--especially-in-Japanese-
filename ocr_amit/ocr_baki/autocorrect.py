def auto_correct(input_ocr):
    ranges = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
        {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
        {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
    ]

    def is_cjk(char):
        return any([_range["from"] <= ord(char) <= _range["to"] for _range in ranges])

    a = input_ocr
    tmp = a
    status = ""
    if "懿" in a:
        for i in range(1, len(a) - 1):
            if a[i] == "懿" and not is_cjk(a[i + 1]) and not is_cjk(a[i - 1]):
                a = a[:i] + "" + a[i + 1:]
    for i in range(0, len(a) - 1):
        if a[i] in ["("] and is_cjk(a[i + 1]):
            status = "japan"
        elif a[i] in ["("] and not is_cjk(a[i + 1]):
            status = "latin"
        elif a[i] in [":"] and is_cjk(a[0]):
            status = "japan:"
        elif a[i] in [":"] and not is_cjk(a[0]):
            status = "latin:"
        if status == "latin":
            for j in range(i, len(a) - 1):
                if a[j] in [")"]:
                    if a[j + 1] != " " and i == 0:
                        tmp = a[:j + 1] + " " + a[j + 1:]
                        break
                    elif a[j + 1] != " " and a[i - 1] == " ":
                        tmp = a[:j + 1] + " " + a[j + 1:]
                        break
                    elif a[j + 1] != " " and a[i - 1] != " ":
                        tmp = a[:i] + " " + a[i:j + 1] + " " + a[j + 1:]
                        break
        elif status == "japan":
            for j in range(i, len(a) - 2):
                if a[j] in [")"]:
                    if a[j + 1] == " " and i == 0:
                        tmp = a[:j + 1] + "" + a[j + 2:]
                        break
                    elif a[j + 1] == " " and a[i - 1] != " ":
                        tmp = a[:j + 1] + "" + a[j + 2:]
                        break
                    elif a[j + 1] == " " and a[i - 1] == " ":
                        tmp = a[:i - 1] + "" + a[i:j + 1] + "" + a[j + 2:]
                        break
        elif status == "latin:":
            for j in range(i, len(a) - 1):
                if a[j] in [":"]:
                    if a[j + 1] != " ":
                        tmp = a[:j + 1] + " " + a[j + 1:]
        elif status == "japan:":
            for j in range(i, len(a) - 2):
                if a[j] in [":"]:
                    if a[j + 1] == " ":
                        tmp = a[:j + 1] + "" + a[j + 2:]
        status = ""
    return tmp
