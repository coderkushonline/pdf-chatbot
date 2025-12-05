from conversation import GetPdfChatBot

chatbot = GetPdfChatBot(filepath="mypdf.pdf")

while True:
    uip = input("Enter your query: ")
    if uip.lower == "exit":
        break
    print(chatbot.qna(uip))
        