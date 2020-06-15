# coding="utf-8"

import numpy as np

Parent_File_Path="C:/ModelWorkPath/corpus_60000/"
Split_Pos_Txt_File_Path=Parent_File_Path+"Ad_Split_Pos_Txt.txt"
Split_Neg_Txt_File_Path=Parent_File_Path+"Ad_Split_Neg_Txt.txt"

Pos_Txt_Index_File_Path=Parent_File_Path+"Pos_Txt_Index.npy"
Neg_Txt_Index_File_Path=Parent_File_Path+"Neg_Txt_Index.npy"

Total_number=60000

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
corpusWord2Vect="C:/PyWorkSpace/FangChuan/myWordVec/w2v_chisim.300d.txt"

allVocabListNPYPath = "C:/PyWorkSpace/FangChuan/myWordVec/allVocabList.npy"
allVocabList = list(np.load(allVocabListNPYPath))

if __name__=="__main__":

    model = Word2Vec.load(corpusWord2Vect)

    Split_Pos_Txt_File = open(Split_Pos_Txt_File_Path,"r",encoding="utf-8")
    Split_Neg_Txt_File = open(Split_Neg_Txt_File_Path,"r",encoding="utf-8")

    Pos_List=[]
    Neg_List=[]

    for i in range(Total_number):
        Pos_List.append(Split_Pos_Txt_File.readline())
        Neg_List.append(Split_Neg_Txt_File.readline())

    Pos_Index_List=[]
    for i in range(Total_number):
        The_Pos_Sample = Pos_List[i][0:-1]    # Remove "\n"
        The_Pos_Sample_List = The_Pos_Sample.split(" ")
        templist = []
        for j in range(len(The_Pos_Sample_List)):
            if The_Pos_Sample_List[j] in model.wv.vocab:
                templist.append(allVocabList.index(The_Pos_Sample_List[j]))

        if len(templist)==0:
            templist.append(0)     # process null
        print(i,"P")
        Pos_Index_List.append(templist)

    Neg_Index_List = []
    for i in range(Total_number):
        The_Neg_Sample = Neg_List[i][0:-1]  # Remove "\n"
        The_Neg_Sample_List = The_Neg_Sample.split(" ")
        templist = []
        for j in range(len(The_Neg_Sample_List)):
            if The_Neg_Sample_List[j] in model.wv.vocab:
                templist.append(allVocabList.index(The_Neg_Sample_List[j]))

        if len(templist) == 0:
            templist.append(0)  # process null
        print(i,"N")
        Neg_Index_List.append(templist)

    # Run one time
    # np.save(Pos_Txt_Index_File_Path,Pos_Index_List)
    # np.save(Neg_Txt_Index_File_Path,Neg_Index_List)


    # print(model.vocab[1])

    # if "/" in model.vocab:
    #     print(np.where(model.vocab=="/")[0][0])
    #     print(model.vocab[np.where(model.vocab == "/")[0][0]])
    # else:
    #     print("null")