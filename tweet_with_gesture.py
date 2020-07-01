import os
import sys
import cv2
import time
import math
import tweepy
import random
import pickle
import credentials
import numpy as np

class HandGesture:
    def __init__(self):
        self.last_gesture = 'nothing'
        self.curr_time = 0
        self.min_time = 30
        self.api = self.get_api()
        self.tweets = self.load_tweets()

    def get_api(self):
        api_key = credentials.api_key 
        api_secret = credentials.api_secret 
        access_token = credentials.access_token
        access_token_secret = credentials.access_token_secret

        auth = tweepy.OAuthHandler(api_key, api_secret) 
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        return api

    def add_tweet(self, gesture, tweet):
        if gesture not in self.tweets:
            self.tweets[gesture] = [tweet]
        else:
            self.tweets[gesture].append(tweet)

    def remove_tweet(self, gesture, tweet):
        if gesture not in self.tweets or tweet not in self.tweets[gesture]:
            return

        self.tweets[gesture].remove(tweet)

    def save_tweets(self, path='tweets.pickle'):
        with open(path, 'wb') as file:
            pickle.dump(self.tweets, file, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved')
           
    def load_tweets(self, path='tweets.pickle'):
        if not os.path.exists(path):
            print('No tweets to load')
            return {}
        
        with open(path, 'rb') as file:
            tweets = pickle.load(file)

        return tweets

    def reset_gesture(self):
        self.gesture = 'nothing'
        self.curr_time = 0

    def publish_tweet(self, gesture):
        print('FOUND ', gesture)

        if gesture not in self.tweets:
            print('Nothing to tweet, try adding a tweet for this gesture')
            return

        try:
            tweet = random.choice(self.tweets[gesture])
            self.api.update_status(tweet)
            print('Tweeted', tweet)
        except Exception as e:
            print(str(e))
            return

    def count_gesture(self, gesture):
        print(gesture)
        if (self.last_gesture == gesture):
            self.curr_time += 1
        else:
            self.last_gesture = gesture
            self.curr_time = 1
        
        if self.curr_time > self.min_time:
            self.curr_time = 0
            self.publish_tweet(gesture)
            return True
        
        return False

    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print('Error opening the video')
            return

        while(cap.isOpened()):
            try:
                ret, frame = cap.read()
                if (ret == False):
                    print('Algo de errado não ta certo')
                    continue

                frame = cv2.flip(frame, 1)
                kernel = np.ones((3,3), np.uint8)
                region_of_interest = frame[100:300, 300:500]
                
                cv2.rectangle(frame, (300,100), (500,300), (0,255,0), 0)    
                hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)
                
                lower_mask = np.array([0,60,100], dtype=np.uint8)
                upper_mask = np.array([20,255,255], dtype=np.uint8)

                mask = cv2.inRange(hsv, lower_mask, upper_mask)                
                mask = cv2.dilate(mask, kernel, iterations = 4)                
                mask = cv2.GaussianBlur(mask, (5,5), 100) 
                
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cnt = max(contours, key = lambda x: cv2.contourArea(x))
                epsilon = 0.0005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                hull = cv2.convexHull(cnt)
                
                areahull = cv2.contourArea(hull)
                areacnt = cv2.contourArea(cnt)
                
                arearatio=((areahull-areacnt)/areacnt)*100

                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)                
                
                if defects is None: continue

                num_points = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    pt = (100,180)
                    
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    s = (a+b+c)/2
                    ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                    
                    d = (2*ar) / a
                    
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                
                    if angle <= 90 and d>30:
                        num_points += 1
                        cv2.circle(region_of_interest, far, 3, [255,0,0], -1)
                    
                    cv2.line(region_of_interest, start, end, [0,255,0], 2)
                                
                num_points += 1
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                if num_points == 1:
                    if areacnt < 2000:
                        self.reset_gesture()
                        cv2.putText(frame,'Put your hand in the box', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    elif arearatio < 12:
                        self.reset_gesture()
                        cv2.putText(frame, '0', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        self.count_gesture('1')
                        cv2.putText(frame, '1', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                            
                elif num_points == 2:
                    self.count_gesture('2')
                    cv2.putText(frame, '2', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                elif num_points == 3:
                    self.count_gesture('3')
                    cv2.putText(frame, '3', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                            
                elif num_points == 4:
                    self.count_gesture('4')
                    cv2.putText(frame, '4', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                elif num_points == 5:
                    self.count_gesture('5')
                    cv2.putText(frame, '5', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                elif num_points == 6:
                    self.reset_gesture()
                    cv2.putText(frame, 'reposition', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                else:
                    self.reset_gesture()
                    cv2.putText(frame, 'reposition', (10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
                cv2.imshow('mask', mask)
                cv2.imshow('frame', frame)
                cv2.waitKey(10)

            except KeyboardInterrupt:
                print("Ok ok, quitting")
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(1)

            except Exception as e:
                print(str(e))
                pass
                    
        cv2.destroyAllWindows()
        cap.release()

def get_menu():
    print('---- Options ----')
    print('1 - Add new tweet')
    print('2 - Remove existing tweet')
    print('3 - Save tweets')
    print('4 - See current tweets')
    print('5 - Run')
    print('6 - Quit')

if __name__ == '__main__':
    get_menu()
    hg = HandGesture()
    
    op = input()

    while op != '6':
        if op == '1':
            gesture = input('Enter gesture: ')
            tweet = input('Enter tweet: ')
            hg.add_tweet(gesture, tweet)
            print('Done')
        elif op == '2':
            gesture = input('Enter gesture: ')
            tweet = input('Enter tweet: ')
            hg.remove_tweet(gesture, tweet)
            print('Done')
        elif op == '3':
            hg.save_tweets()
        elif op == '4':
            print(hg.tweets)
        elif op == '5':
            hg.run()
        else: 
            print('Invalid operation')

        op = input()