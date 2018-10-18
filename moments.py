import cv2 as cv


def testMatch(original, templ):
    src = original.copy()
    rgb = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    rgbT = cv.cvtColor(templ, cv.COLOR_GRAY2BGR)

    # получаем границы изображения и шаблона
    srcCanny = cv.Canny(src, 50, 200)  # binI
    templCanny = cv.Canny(templ, 50, 200)  # binT

    # находим контуры изображения
    _, contoursI, hierarchyI = cv.findContours(srcCanny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # находим контуры шаблона
    _, contoursT, hierarchyT = cv.findContours(templCanny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    seqT = 0
    perimT = 0
    # находим самый длинный контур
    for seq0 in range(len(contoursT)):
        perim = len(contoursT[seq0])
        if perim > perimT:
            perimT = perim
            seqT = seq0

    cv.drawContours(rgbT, contoursT, seqT, (0, 255, 0), 1, cv.LINE_8, hierarchyT, 0)
    cv.imshow("Template Contours", rgbT)

    seqM = 0
    ret = 0.1

    # обходим контуры изображения

    for seq0 in range(len(contoursI)):
        match0 = cv.matchShapes(contoursI[seq0], contoursT[seqT], cv.CONTOURS_MATCH_I3, 0.0)
        if match0 <= ret:
            cv.drawContours(rgb, contoursI, seq0, (0, 266, 0), 1, cv.LINE_8, hierarchyI, 0)
            # print("TRUE: ", match0)
        # print("match: ", match0)

    cv.imshow("find", rgb)

    cv.waitKey(0)


def main():
    fname = "imgs/moments.png"
    fnameT = "imgs/templ.png"

    original = cv.imread(fname, 0)
    tpl = cv.imread(fnameT, 0)

    testMatch(original, tpl)


main()
