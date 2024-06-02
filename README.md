**Zomato Sales Data**

In this data set, I have analyzed the Zomato sales data for Bangalore city. In this analysis, I have aimed to answer the following questions:
1)	Relationship between Online/Offline Orders and Restaurant Ratings:

![image](https://github.com/nikunjbharti/zomato_sales_data/assets/163707111/1f227983-35d9-4be7-8ce3-28db4a5b4557)


  **a)	Low to Mid Ratings (1.8 - 3.7):**
  Restaurants with lower ratings (1.8 - 2.3) mostly do not accept online orders.

  In the mid-rating range (2.4 - 3.7), there is a more balanced mix, but the presence of restaurants accepting online orders increases.

  **b)	High Ratings (3.8 - 4.9):**
  Restaurants with higher ratings (3.8 - 4.9) predominantly accept online orders, with very few exceptions.

  The trend suggests that higher-rated restaurants are more likely to integrate online ordering systems, which could be contributing to their higher ratings.

  **c)	Business Implications:**
  For Lower-Rated Restaurants: Consider integrating online ordering systems to improve ratings and customer satisfaction.

  For Higher-Rated Restaurants: Maintain and enhance online ordering services to sustain high ratings.

2)	Created a word cloud to get an idea about the most famous restaurants in Bangalore city.
![image](https://github.com/nikunjbharti/zomato_sales_data/assets/163707111/c6743280-0cc9-4810-ae3b-e7ee940641cc)
 

3)	Performed bi-gram and tri-gram analysis of customer reviews and concluded the following for Quick Bites restaurant:
![image](https://github.com/nikunjbharti/zomato_sales_data/assets/163707111/c6333fb9-5dcc-4ff7-b53e-f2845b98754d)

![image](https://github.com/nikunjbharti/zomato_sales_data/assets/163707111/9952b69e-3a96-48f2-83c4-dd027824e1ce)

 

  **Popular food items:**
  Some of the most frequent trigrams include "chicken fried rice", "paneer butter masala", "south indian food" and "chicken biryani", indicating these are popular food choices at Quick Bites.

  **Restaurant sentiment:**
  "good food good", "food really good", and "taste really good" suggest positive customer sentiment towards the food quality. Other trigrams like "must try place" and "like home made" indicate a positive dining experience.

  **Value for money:**
  Trigrams like "good value money", "Worth every penny" and "reasonable price" suggest that customers perceive Quick Bites to offer good value for money.

4)	I have automated this code for all the restaurants.

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams
import nltk

def process_reviews(rest_type):
    data = df.dropna(subset=['rest_type'])
    # Filter the data for the specified restaurant type
    r = data[data['rest_type'].str.contains(rest_type)]
    
    # Convert reviews to lowercase
    r['reviews_list'] = r['reviews_list'].apply(lambda x: x.lower())
    
    # Tokenize the reviews
    tokenizer = RegexpTokenizer("[a-zA-Z]+")
    reviews_tokens = r['reviews_list'].apply(tokenizer.tokenize)
    
    # Download the stopwords corpus if not already downloaded
    nltk.download('stopwords', quiet=True)
    
    # Define the stop words
    stop = stopwords.words('english')
    stop.extend(['rated', 'n', 'x', 'X', 'RATED', 'Rated', 'nan', 'N', 'NAN'])
    
    # Remove stop words from the tokens
    reviews_tokens_clean = reviews_tokens.apply(lambda x: [token for token in x if token not in stop])
    
    # Flatten the list of tokens for bigrams and trigrams analysis
    total_reviews_1D = [word for sublist in reviews_tokens_clean for word in sublist]
    
    # Calculate bigrams and their frequencies
    bi_gram = bigrams(total_reviews_1D)
    fd_bigrams = FreqDist(bi_gram)
    
    # Calculate trigrams and their frequencies
    tri_gram = trigrams(total_reviews_1D)
    fd_trigrams = FreqDist(tri_gram)
    
    # Plot the top 30 bigrams and trigrams with dynamic titles
    plt.figure(figsize=(20, 10))
    fd_bigrams.plot(45, title=f"Top 30 Bigrams of {rest_type} Restaurant")
    plt.figure(figsize=(20, 10))
    fd_trigrams.plot(45, title=f"Top 30 Trigrams of {rest_type} Restaurant")

# Example usage:
# process_reviews('desired_restaurant_type')
