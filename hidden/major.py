lines_gold = open("./data/Arapaho/arp-test-track1-uncovered", "rb").readlines()
lines_gold = [line_gold.decode(errors='ignore').strip() for line_gold in lines_gold]

lines = open("./predictions/arp_track1_morph_trial1.prediction", "rb").readlines()
lines = [line.decode(errors='ignore').strip() for line in lines]

lines2 = open("./predictions/arp_track1_morph_trial2.prediction", "rb").readlines()
lines2 = [line2.decode(errors='ignore').strip() for line2 in lines2]

lines3 = open("./predictions/arp_track1_morph_trial3.prediction", "rb").readlines()
lines3 = [line3.decode(errors='ignore').strip() for line3 in lines3]

lines4 = open("./predictions/arp_track1_morph_trial4.prediction", "rb").readlines()
lines4 = [line4.decode(errors='ignore').strip() for line4 in lines4]

lines5 = open("./predictions/arp_track1_morph_trial5.prediction", "rb").readlines()
lines5 = [line5.decode(errors='ignore').strip() for line5 in lines5]

lines6 = open("./predictions/arp_track1_morph_trial6.prediction", "rb").readlines()
lines6 = [line6.decode(errors='ignore').strip() for line6 in lines6]

lines7 = open("./predictions/arp_track1_morph_trial7.prediction", "rb").readlines()
lines7 = [line7.decode(errors='ignore').strip() for line7 in lines7]

lines8 = open("./predictions/arp_track1_morph_trial8.prediction", "rb").readlines()
lines8 = [line8.decode(errors='ignore').strip() for line8 in lines8]

lines9 = open("./predictions/arp_track1_morph_trial9.prediction", "rb").readlines()
lines9 = [line9.decode(errors='ignore').strip() for line9 in lines9]

lines10 = open("./predictions/arp_track1_morph_trial10.prediction", "rb").readlines()
lines10 = [line10.decode(errors='ignore').strip() for line10 in lines10]

from collections import Counter


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

major_pred = open('majority_Arapaho_test_hidden.txt','w')
for i, line_gold in enumerate(lines_gold):
    major_set_gls = []
    if line_gold.startswith('\g'):
        sent_gloss_gold = line_gold.split(' ')
        sent_gloss1 = lines[i].split(' ')
        sent_gloss2 = lines2[i].split(' ')
        sent_gloss3 = lines3[i].split(' ')
        sent_gloss4 = lines4[i].split(' ')
        sent_gloss5 = lines5[i].split(' ')
        sent_gloss6 = lines6[i].split(' ')
        sent_gloss7 = lines7[i].split(' ')
        sent_gloss8 = lines8[i].split(' ')
        sent_gloss9 = lines9[i].split(' ')
        sent_gloss10 = lines10[i].split(' ')
        #print(sent_gloss9)
        current_gls = []
        for num in range(len(sent_gloss_gold)):
            each_all_gls = []
            #print(sent_gloss1[num])
            #print(sent_gloss9[num])
            if len(sent_gloss1) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss1[num])
            #else:
             #  each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss2) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss2[num])
            #else:
             #  each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss3) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss3[num])
            #else:
             #  each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss4) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss4[num])
            #else:
            #   each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss5) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss5[num])
            #else:
            #   each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss6) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss6[num])
            #else:
            #   each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss7) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss7[num])
            #else:
            #   each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss8) >= len(sent_gloss_gold):
               each_all_gls.append(sent_gloss8[num])
            #else:
             #  each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss9) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss9[num])
            #else:
             #  each_all_gls.append(sent_gloss_gold[num])
            if len(sent_gloss10) >= len(sent_gloss_gold) :
               each_all_gls.append(sent_gloss10[num])
            #else:
              # each_all_gls.append(sent_gloss_gold[num])
            major_gls = Most_Common(each_all_gls)
            #print(major_gls)
            current_gls.append(major_gls)
        #print(current_gls)
        major_pred.write(' '.join(e for e in current_gls)+'\n')
    else:
        major_pred.write(line_gold+'\n')


