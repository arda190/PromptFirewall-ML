import math
from my_LogisticRegression import LogisticRegression




class TFIDF:
    def __init__(self):
        self.stopWords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'could',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'a', 'an', 'the', 'and', 'but', 'or', 'as', 'for', 'nor', 'on', 'at', 'by',
    'with', 'without', 'of', 'off', 'over', 'under', 'above', 'below', 'between',
    'among', 'through', 'during', 'within', 'without', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'since', 'until',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'but', 'per', 'via', 'vs', 'etc', 'e.g.', 'i.e.',
    # Common additional ones
    'could', 'would', 'should', 'might', 'must', 'shall', 'will', 'may', 'can',
    'done', 'going', 'get', 'got', 'gotten', 'make', 'made', 'see', 'seen',
    'know', 'known', 'think', 'thought', 'want', 'say', 'said', 'tell', 'told'
    }
        self.idfValues = {}
        self.vocabulary = {}





    def lower_all(self,x_):
       return x_.lower()


    def remove_punctuation(self,x_):
        punctuation = ['.',',','!','?']

        for i in punctuation :
            if i in x_ :
               x_ = x_.replace(i,"")

        return x_


    def remove_stop_words(self,x_):

            new_words = []

            for word in x_.split():
                if word not in self.stopWords :
                    new_words.append(word)

            return " ".join(new_words)



    def process_text(self,data):

        processed_data = []

        for x in data :
            x = self.lower_all(x)
            x = self.remove_punctuation(x)
            x = self.remove_stop_words(x)
            processed_data.append(x)

        return processed_data


    def calculate_IDF(self,data,doc_count):

        idf = {}

        for word in data.keys() :
            idf[word] = math.log(doc_count/data[word])

        return idf




    def createIDF(self,data):

        idf_values = {}

        for sentence in data :
            unique_words = set(sentence.split())

            for word in unique_words :
                idf_values[word] = idf_values.get(word,0)+1


        self.idfValues = self.calculate_IDF(idf_values,len(data))



    def createVocabulary(self,data):

        vocabulary = {}

        for sentence in data :
            for word in sentence.split() :
                if word not in vocabulary :
                   vocabulary[word] = len(vocabulary)

        self.vocabulary = vocabulary






    def fit(self,data):
        processed_date = self.process_text(data)

        self.createVocabulary(processed_date)
        self.createIDF(processed_date)



    def transform(self,data):

       data = self.process_text(data)

       transformed_data = []

       for sentence in data:
           word_count = {}
           total_words = len(sentence.split())
           feature = [0] * len(self.vocabulary)
           for word in sentence.split() :
               word_count[word] = word_count.get(word,0)+1


           for w in word_count.keys() :
               if w in self.vocabulary :
                  tf = word_count[w]/total_words
                  idf = self.idfValues[w]
                  p = tf * idf

                  index =self.vocabulary[w]

                  feature[index] = p
           transformed_data.append(feature)


       return transformed_data





    def fit_transform(self,data):
        self.fit(data)

        return self.transform(data)



model = TFIDF()

documents = [
    "Win a free iPhone", "Meeting tomorrow at 10", "Claim your free prize", "Let's have lunch",

    # --- Spam (1) ---
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
    "URGENT: Your account has been compromised. Log in immediately to verify your identity.",
    "Get rich quick! Earn up to $5000 a week working from the comfort of your home.",
    "FREE entry into our annual cash prize draw! Reply with your details to enter now.",
    "You have been selected for a exclusive private loan offer. Low interest rates guaranteed.",
    "Double your crypto portfolio overnight! Join our premium trading signals channel today.",
    "FINAL NOTICE: Your IRS tax refund is waiting for approval. Submit your form now.",
    "Get the body you always wanted with our all-natural weight loss miracle pills.",
    "Your package could not be delivered due to an incorrect address. Update details here.",
    "Act fast! 90% off all luxury watches for the next 2 hours only. Buy now.",
    "Dear customer, your bank card has been suspended. Please update your security PIN.",
    "Congratulations! Your phone number won the 2nd prize in our international lottery.",
    "Earn cash just by watching videos! No experience required. Sign up instantly.",
    "Exclusive offer: Save big on medical prescriptions. No insurance needed.",
    "You qualify for a free government grant up to $10,000. Click to apply.",
    "Hot singles in your neighborhood want to meet you tonight! Create a free profile.",
    "Refinance your mortgage today and skip your next two monthly payments completely.",
    "Your online order is ready, but we need your credit card billing details to ship.",
    "Congratulations to our lucky winner! You have been awarded a brand new iPad Pro.",
    "Unsecured personal lines of credit up to $50k. No credit check required to qualify.",
    "Improve your website SEO ranking to page 1 guaranteed in less than 30 days.",
    "Urgent notification regarding your auto warranty expiration. Renew today for cheap.",
    "You have a pending payout of $450.50 from your loyalty points. Cash out here.",
    "Get cheap insurance quotes in seconds. Save hundreds on your annual premium.",
    "Congratulations! You are today's lucky visitor. Click to claim your mystery prize.",
    "Make $500 daily by simply driving your car with our company logo sticker.",
    "Shocking discovery reveals how to reverse aging naturally. Watch the free video.",
    "Your credit score is dropping! Sign up for our repair service to fix it now.",
    "Free hotel stay vouchers available for the first 50 respondents. Reply YES.",
    "Invest in the next big tech startup before it goes public. Huge returns expected.",
    "ALERT: Suspicious login attempt detected on your streaming account. Reset password.",
    "Get your degree online in just 6 months. High paying jobs waiting for you.",
    "Congratulations! You won a free cruise to the Bahamas. Call this number now.",
    "Unlock hidden features on your phone and get free unlimited data forever.",
    "Need cash fast? Get approved for a payday loan in less than 5 minutes.",
    "Exclusive luxury perfume replicas at a fraction of the cost. Shop the clearance.",
    "Your subscription will automatically renew at $499. Cancel now if this was an error.",
    "Earn passive income while you sleep using this automated AI trading software.",
    "Congratulations! You have been chosen to receive a free sample box of cosmetics.",
    "Clearance sale on all designer shoes. Buy one get two free ends tonight.",
    "Urgent security alert: Your cloud storage is full. Buy extra space to save files.",
    "You are eligible for a student loan forgiveness program. Call our advisors now.",
    "Get a free background check on anyone instantly. 100% anonymous search tool.",
    "Congratulations! Your resume was selected for an executive remote position.",
    "Special discount on generic medication. Fast shipping straight to your doorstep.",
    "Earn money by testing new mobile applications on your phone. Limited slots.",
    "Your invoice #8493 is overdue. Please pay immediately to avoid late fees.",
    "Congratulations! You have been chosen as our user of the day. Claim rewards.",

    # --- Ham (0) ---
    "Can you send over the updated project notes before the deadline?",
    "Hey, are we still on for coffee this afternoon at the usual place?",
    "Thanks for sending the report, I will review it first thing tomorrow.",
    "Let me know what time works best for you to hop on a quick call.",
    "Hi mom, just calling to see how you are doing. Talk to you later.",
    "Please review the attached invoice and let me know if it looks correct.",
    "Are you available to help me move some furniture this coming Saturday?",
    "The team meeting has been rescheduled to Thursday morning at nine.",
    "Hey, did you leave your jacket in my car after the movie last night?",
    "I'll be a few minutes late to dinner, please go ahead and order.",
    "Can you double check the spreadsheet to ensure the math adds up?",
    "Just wanted to wish you a very happy birthday! Hope you have a great day.",
    "Please remember to pick up some milk and eggs on your way home.",
    "Hi, I am interested in renting the apartment you listed on the website.",
    "The doctor called to confirm your annual checkup appointment next week.",
    "Could you please forward me the contact information for the contractor?",
    "Hey, the dog needs to go out for a walk before it gets dark.",
    "Thank you for the wonderful dinner last night, we had a fantastic time.",
    "Don't forget that the library books are due back by this Friday.",
    "Can you recommend a good mechanic in the area for a quick oil change?",
    "Hi, just following up on my job application submitted last Tuesday.",
    "The kids are ready to be picked up from soccer practice whenever you are.",
    "Please find attached the slides for our presentation next Monday.",
    "Hey, do you want to grab tickets for the game this weekend?",
    "I left the keys on the kitchen counter next to the microwave.",
    "Can we reschedule our lunch date to next week? Something came up.",
    "Hi, I noticed a typo on the third page of the proposal document.",
    "The conference call details have been sent to your calendar invite.",
    "Hey, did you get a chance to read that article I sent you yesterday?",
    "Please make sure to lock the back door before you go to bed.",
    "Thank you for helping me study for the exam, I really appreciate it.",
    "Can you send me the address of the restaurant for tonight's party?",
    "Hi, I will be out of the office starting tomorrow until next Monday.",
    "The landlord said he will come by tomorrow to fix the leaky faucet.",
    "Hey, let's plan a weekend trip sometime next month if you're free.",
    "Please print out a copy of the agenda for everyone attending the meeting.",
    "Could you please check if the mail has arrived yet today?",
    "Hi, I wanted to check if you received the package I mailed out.",
    "The project deadline has been extended by a week, so we have time.",
    "Hey, do you know where I can find the instructions for the lawnmower?",
    "Please let me know when the item is back in stock at the store.",
    "Thank you for the quick response, that clarifies everything perfectly.",
    "Can you pick up the dry cleaning sometime before six this evening?",
    "Hi, I am writing to request a copy of my official academic transcript.",
    "The weather forecast says it might rain later, so bring an umbrella.",
    "Hey, are you using the conference room right now or is it empty?",
    "Please review these design mockups and give me your honest feedback.",
    "Could you drop me off at the train station tomorrow morning early?"
]

labels = [
    # --- Original Labels ---
    1, 0, 1, 0,

    # --- Spam Labels (48 items) ---
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,

    # --- Ham Labels (48 items) ---
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
]

X = model.fit_transform(documents)

logisticRegression_model = LogisticRegression()


logisticRegression_model.fit(X, labels)


email=model.transform(["urgent security warning"])

print(logisticRegression_model.predict(email[0]))

print(logisticRegression_model.weights)
print(logisticRegression_model.bias)













