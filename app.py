from flask import Flask, render_template, request,url_for
import requests
from train import lemmatizerfun
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import urljoin
import pandas as pd
from urllib.parse import urlparse
app = Flask(__name__)
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


with open("vectorizer.pkl","rb") as vectorizerfile:
   vectorizer=pickle.load(vectorizerfile)

analyzer=SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review=request.form['review']
    cleaned_review=lemmatizerfun([review])
    train_x_vector = vectorizer.transform(cleaned_review).toarray()
    prediction=model.predict(train_x_vector)
    if(prediction==1):
        result='Original Review'
    else:
        result='Fake Review'
    return render_template('result.html',review=review,prediction=prediction)
def webscrapping(url):
    parsed_url=urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
       return None,None,None,"Invalid URL. Please enter a valid product URL"
    Headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0',
        'Accept-Language': 'ven-US,en;q=0.9,en-IN;q=0.8',
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Connection": "keep-alive",
        'Cookie':'session-id=260-0592466-5030314; i18n-prefs=INR; ubid-acbin=260-6056319-7689548; lc-acbin=en_IN; JSESSIONID=591E0D9B1EB3E60D69B87F1321E52023; at-acbin=Atza|IwEBIKvQg1MkXM6x__CDERxss_PI2yDYUpewOOTawI7WCfy6WT3Z_S6KR09nFjCKcCaPhs3PMmJlnMVx9fXKYlvmpUuvSxe56k2Lep0abrnI78WdwVEYdqQ36kcxgJNkmhbDQPw1w4di6_OU0vl9IpdNt8zGijD68DDMJey1ev7uYjL2PAqfERYRthBw4bCi1XUayXrCjQ3n9vDsVzJ-XvpcNf3_FvCWI9vlAYJ9auEiEiPpGQ; sess-at-acbin="77VBuhtpBQmQckR7Pqmr1//9u0kRTZJ0ew6vdMEt2Vo="; sst-acbin=Sst1|PQFUzab8rd7LwHFBCpwIg1ZYCWQ8O7gKJOqCrjHCMNQP2QC7AvB8Hq3pO2QixvAENm_eaceGfVmO1wY3uwvPtzy-TlQ7tsgq4RnUf7W6pDx5XRfE5jDBGijGTnljg2Ynyg6Q8Lw6-Xt23-SVw6hb2kRpH0Wc16ov70V4sg1ZId9PDfKF8DxI6fquv8IyC1GvDCOLp63xrx8SxMPwj5RCDui5elJgqQKLEhhm5887yKtmqH_tS-IcvbPX1laW8SEuFKE4zKh_q4r5mE7uMVEA_tJFWmw_oE-x5UB6h-svNpjRpNU; session-id-time=2082787201l; session-token=b0BQ4pJ315cxoblyfL+4lrQDEL9Z+m6P2mcgj9mQdruo7uAaTtT4ve/jmdBjqVwckgqxDobNzCUnz/4xoti5zttDxZdmInKcABp1z0B7i1SaBKMPy/8zJUryHkitrOuQYO9QwHLSSK9tKBSseUd129WZRYS18msdcBkQ7vpsLj+RHvyTR3tPJjGiTUS1kTbShUVAjZj89Ehrci0wKZsun2lf4n5sMmClcthXPdef+UKfN5t2/tRPXLRmWH5LXZ4cJx7b+WEA4zrnBWLfacwg/uYkmcUyFmX1xngTwKKo9m/RZJ+6cl9otMw5xETfKbpNDQCy1DPeTBx37cx4KvIaHTxgbNkmT+rU/w7DpnZjPsfnBA2PVgVDHOU33aVRDebd; x-acbin="ieIWZ2dQuJXzPG145ATV7J4oprg2vUUWxUzCT0zd2zHs9kTL5W?HowLuhKrvEUTy"; csm-hit=tb:s-SFXD0N7FRKCKH15MQXZV|1736668537656&t:1736668537656&adb:adblk_no'

    }
    try:
        all_reviews=[]
        productpage=requests.get(url,headers=Headers)
        if productpage.status_code!=200:
            return None,None,None,"Failed to fetch product page.Please check the URL"
        if productpage.status_code==200:
           soup=BeautifulSoup(productpage.text,'html.parser')
           reviewurl=soup.find('a', attrs={'data-hook': 'see-all-reviews-link-foot', 'class': 'a-link-emphasis a-text-bold'})
           if not reviewurl:
               return None,None,None,"No reviews found for this product"
           reviewurl = urljoin(url,reviewurl['href'])
           img=soup.find(class_='imgTagWrapper')
           imgurl=img.find('img')['src']
           title=soup.find('span',{'id':'productTitle'}).text.strip()
        for n in range(1,5):
            url1=reviewurl.replace("cm_cr_dp_d_show_all_btm",f"cm_cr_arp_d_paging_btm_next_{n}") + f"&pageNumber={n}"
            page=requests.get(url1,headers=Headers)
            if page.status_code==200:
                soup=BeautifulSoup(page.text,'html.parser')
                reviews = soup.find_all('span', {'data-hook': 'review-body'},class_="a-size-base review-text review-text-content")
                for review in reviews:
                    all_reviews.append(review.get_text().strip())
            else:
                break
        if not all_reviews:
            return None,None,None,"No review found dor this product"
        return all_reviews,imgurl,title,None
    except Exception as e:
        return None,None,None, f"An error occured :{str(e)}"
       
@app.route('/scrape', methods=['POST'])
def scrape():
    url=request.form['url']
    all_reviews,imgurl,title,error=webscrapping(url)
    if error:
        return render_template('result.html',error=error)
    all_reviews_df=pd.DataFrame({'reviews':all_reviews})
    cleaned_reviews=lemmatizerfun(all_reviews_df['reviews'].values)
    train_x_vector=vectorizer.transform(cleaned_reviews).toarray()
    prediction=model.predict(train_x_vector)
    fake_count=sum(prediction==0)
    fake_percent=int((fake_count/len(prediction))*100)
    original_percent=int(100-fake_percent)
    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for review in all_reviews:
        sentiment = analyse_setiment(review)
        if sentiment['compound'] >= 0.05:
            sentiment_scores['Positive'] += 1
        elif sentiment['compound'] <= -0.05:
            sentiment_scores['Negative'] += 1
        else:
            sentiment_scores['Neutral'] += 1
    barchart_path = plot_barchart(["Fake", "Original"], [fake_percent, original_percent])
    piechart_path = plot_piechart(sentiment_scores)
    return render_template("result.html",title=title,imgurl=imgurl,fake_percent=fake_percent,original_percent=original_percent,barchart_path=barchart_path,piechart_path=piechart_path)
def plot_barchart(labels,values):
   plt.figure(figsize=(6,4))
   plt.bar(labels,values,color=["red","green"])
   plt.title("Fake vs Original Reviews")
   plt.xlabel("Type")
   plt.ylabel("Percentage")
   barchart_path = "static/barchart.png"
   plt.savefig(barchart_path)
   plt.close()
   return barchart_path
def plot_piechart(data):
    plt.figure(figsize=(6, 6))
    plt.pie(data.values(), labels=data.keys(), autopct="%1.1f%%", colors=["blue", "orange", "gray"])
    plt.title("Sentiment Analysis")
    piechart_path = "static/piechart.png"
    plt.savefig(piechart_path)
    plt.close()
    return piechart_path
def analyse_setiment(text):
  sentiment_score=analyzer.polarity_scores(text)
  return sentiment_score

if __name__ == '__main__':
    app.run(debug=True)
