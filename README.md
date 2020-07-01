# Tweet With Gesutre

Using OpenCV and Twitter API to post tweets with hand gestures. Each gesture is releated to a list of possible tweets and once the gesture is recognized, a random tweet of its list is published.

## Requirements

You can install all the necessary modules using the requirements file as follows:

```bash
pip3 install -r requirements.txt
```

You will also need to get credentials to access the twitter API and, after that, create a python file named credentials.py in which the keys must be stored as strings.

## How to use

Once you run the code, you have a few options:

- Add new tweet: inform a gesture and the tweet, and this tweet will be associated with the gesture
- Remove existing tweet: remove tweet from a gesture's list
- Save tweets: stores the current tweets in pickle file, so they can be reused
- See current tweets
- Run
- Quit