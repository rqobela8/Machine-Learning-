import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re
from message import Message




def tokenize(text:str)->Set[str]:
    """
        Tokenize text

        Args:
            text (str): Original


        Returns:
            List[str]: each word that appears in text
        """
    text = text.lower()
    all_words = re.findall("[a-z0-9]+",text)

    return set(all_words)



class NaiveBayesClassifier:
    def __init__(self,k:float) -> None :
        """
        Constructor

        Attributes
            k:float : pseudocunter
            spam_messages:int : Counter for spam messages
            ham_messages:int : Counter for non spam messages
            spam_token_counts:Dict[str,int] : How many times word(key) appears in spam message
            ham_token_counts: Dict[str,int] : How many times word(key) appears in non-spam message
            tokens:Set[str] : All the words our model was trained on.

        """
        self.k = k
        self.spam_messages = 0
        self.ham_messages = 0
        self.spam_token_counts : Dict[str,int] = defaultdict(int)
        self.ham_token_counts: Dict[str, int] = defaultdict(int)
        self.tokens: Set[str] = set()

    def train(self,messages:List[Message])->None:
            """
               Create and Train Model On messages

               Args:
                   messages (List): List with Message object data

               Returns:
                   None
            """

            for message in messages:
                if message.is_spam:
                    self.spam_messages+=1
                else:
                    self.ham_messages +=1

                for token in tokenize(message.text):
                    self.tokens.add(token)
                    if message.is_spam:
                        self.spam_token_counts[token]+=1
                    else:
                        self.ham_token_counts[token]+=1

    def _probability(self,token:str)->Tuple[float,float]:
        """
            Calculates P(Xi|Spam) and P(Xi|nonspam)

            :param token: word
            :return: prob_if_spam and prob_if_ham

        """
        spam_count = self.spam_token_counts[token]
        ham_count = self.ham_token_counts[token]

        p_spam_token = (self.k + spam_count) / (2 * self.k + self.spam_messages)
        p_ham_token = (self.k + ham_count) / (2 * self.k + self.ham_messages)

        return p_spam_token, p_ham_token


    def predict(self,message:str)->float:
        """
            Predicts probability of message is spam

            :param message: text
            :return: probability number
        """
        tokens = tokenize(message)

        log_prob_if_spam = 0
        log_prob_if_ham = 0

        for token in self.tokens:
            if (self.ham_token_counts.get(token,0)+
                self.spam_token_counts.get(token,0) )>=2:

                prob_if_spam,prob_if_ham = self._probability(token)

                # add the log probability of seeing it
                if token in tokens:
                    log_prob_if_spam += math.log(prob_if_spam)
                    log_prob_if_ham += math.log(prob_if_ham)

                # Otherwise add the log probability of _not_ seeing it,
                # which is log(1 - probability of seeing it)
                else:
                    log_prob_if_spam += math.log(1.0 - prob_if_spam)
                    log_prob_if_ham += math.log(1.0 - prob_if_ham)



        log_diff = log_prob_if_ham - log_prob_if_spam

        return 1 / (1 + math.exp(log_diff))







