# Scene_classification_using_keras
This is a multi-image classification problem.A simple keras model is used.

Image_classes(preview in github_data folder):

1.buildings.

![alt text](https://storage.googleapis.com/kagglesdsdata/datasets/111880/269359/seg_train/seg_train/buildings/10006.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572433040&Signature=cXV6TvIn7nQbk9BtgJDM5ty4KNhThIQ%2F8kMUnyzvfo0%2FadgsZCa9h7DIWoHHPBGYVeQpcYzjE7N865BTk20JBEvnbE7ib8gCB1hQpGgUJLv5vUwWaKej9wNzz%2FnTS7u4PehU5mKbUK4TpJupXHQPVfN3x5GNDB%2FTFRhABQvOjzietBTokL4E2vtPfSMBskPVLIKJ9ZAvS2mCVvh9AxZgbS7bgJgKI0PMU%2BcivMyIflShVvpNF4wmZwl2DlgmFdzoi8Pfu954njg1nMGJoVOsSS7St%2FwDfWXL6mEQvPb0o33XzzhAPmGAL8Sk3LRgjsLyzqEhF4ozI%2BryRwsbMtPF6Q%3D%3D)

2.forest.

![alt text](https://storage.googleapis.com/kagglesdsdata/datasets/111880/269359/seg_test/seg_test/forest/20056.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572433139&Signature=cvtcLB%2F1uJ%2Bfanfyk6dg4panJW1HWQUXym5iqL%2F5vCcK2gGaW%2FHy0Iw5lQEswYny4V%2Fecm2MM1BCVTfsuu0uA%2B00KuR7j9gpiLf3f%2FE19nlUzAjUtdG76o5t3xizSMC7kc%2FWp3P%2Bfy%2BGUynUGkGuLJC%2BJ1MgYPR6SRqxRQMpGNzriYeKmQJhjorKCoIvjD3SHHlI5r%2F80lNzeal9eBnbAqwP8WVv1wPxiJSNZzSV%2BWgp8Es9Ws76%2FEr4rmPCExwc0zwxOrCQwsz5uvZxwMAaYDt3qkH7%2BGKLFhw%2B8eM2pkmI5mPyMG0%2BXlVeWZByAoqWfYEj5dmB%2Ft23v8pKwNllFg%3D%3D)

3.glacier.

![alt text](https://storage.googleapis.com/kagglesdsdata/datasets/111880/269359/seg_train/seg_train/glacier/10003.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572433227&Signature=rAXsT34Yg8%2B%2BLp6ZqA4G97lwg7DTKJnzelBBPUKS0lpVwbrmRIf3UrzIXKnDZBkLeP2A%2FyO%2F6Zp0ruG7n3s5iC6IPO%2BrQnR1BHFAassl8AZWZsRz3floxMWTzuJmV1%2B2a4rBk3C64BBnsUSuqRQOGnpIGoxKvNHvi8PFvhY0mxuljMPXUFbFC7b9LYIqhVuZWieBHP687GgTCV0TUAabd%2BPWFxr4%2F%2FKzpnzQ7fN7X9XX%2FErDZFuuahA5ZpiMi%2FdAcXjbICmhILn93i1bvcW6WZVF%2FuFDfStMvjUIVOjbcOl0jAiydFKNXsGSY22UHQ1mp9Cg%2BEokdtSHhbXBKNveCg%3D%3D)

4.street.

![alt text](https://storage.googleapis.com/kagglesdsdata/datasets/111880/269359/seg_test/seg_test/street/20075.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572433207&Signature=ZPUjP%2Bs1ReDoA%2FRel8%2BJHzmt98cJBSZXeXNhqzwxheaa3oe%2BIkM7Ws%2FS%2FeQZsc94SEA01rMBCSPTf%2FeyqSV505XH1Ju93S9pUCQ%2BFgjbE39Uv7wedXNSWN8WHPhKgFHdYB4X5BFCUX6Roj5XfMfisQORpCRrVcxZ5w8u%2B25e%2BmdXhBi8O5wCH0s%2BdF9ownpwzBdPmzyfRxqpuXTC26Dr8pWSMU6S0m0H6Esr8JUEc89vav3syW80qdqa43nl%2FPQLgvlCu8SXCyO56G07t3eBbol0ZaUPQX7arGLmsrMHaTJ%2BpPuXxFgAnYLv57E44g%2FMpcOFynCZhjhVMR%2FM9DBp8g%3D%3D)

Each class contains:

**Train**:~950 images of each class.

**Validation**:200 images of each class.

**Test**:images of each class.

Notebook : keras_implementation_small.ipynb contains the implementation code

Results are shown in result folder

**Error metric**: Categorical_crossentropy

Final Accuracy on test images(40): **95%**
