# Model card: H&M weekly recommendations

## What it does
For each customer, it suggests 12 products they might buy in the coming week. A separate search feature lets someone describe what they want in plain words and get matching products back.

## What it's for
This is a portfolio project, not a production system. It shows how I'd build, test and judge a recommender. Nobody should use it to make real business decisions.

## Data
H&M Personalized Fashion Recommendations on Kaggle, which is licensed for non-commercial use. To fit a 16 GB laptop with no GPU, I used a random 20% of customers (274,259 people) and only their last 26 weeks of purchases, from 25 March to 22 September 2020 (1.6 million rows). I checked that the sample matched the full customer table on age, shopping frequency and the share of one-time buyers.

## How it's built
1. **Shortlist:** for each customer, about 50 candidate items from their own recent purchases, items often bought together with those, items with similar text descriptions, and current best-sellers.
2. **Ranking:** a LightGBM model orders the shortlist using 14 features, such as how recently the customer bought that item, how popular it is right now, and how often it's bought together with their past items.
3. **New customers:** best-sellers from the last 2 weeks.
4. **Search:** a sentence-transformer finds the 20 closest products to the request, then Gemini picks the best matches from only those 20.

## How I tested it
I split the data by time, never at random. I tuned everything on one validation week (9 to 15 September 2020) and scored the final test week (16 to 22 September) only once, at the end.

| Method | Test MAP@12 |
|---|---|
| Popularity (last 2 weeks) | 0.00794 |
| Past items (last 8 weeks) | 0.02412 |
| + bought together | 0.02472 |
| **LightGBM ranker** | **0.02818** |

The ranker beat my best simple method by 14%, with a 95% bootstrap range of +0.00232 to +0.00472. A text-embedding method and a two-tower neural network didn't beat the simpler methods on validation, so I left them out of the final test.

For search, Gemini's picks were correct 95% of the time on 14 test requests, compared with 0.77 precision@5 for retrieval alone. Every recommended product came from the retrieved list, and it correctly recommended nothing for 2 requests H&M can't serve.

## Limits
- **Low scores overall.** Only about 8% of real purchases were in the shortlist, so most customers can't get a correct item no matter how well the model ranks. A better shortlist would probably help more than a better model.
- **Optimistic test week.** My weekly check showed the validation and test weeks were among the best of the 13 I looked at. The simple method's test score was about 14% above its 13-week average, so a typical week would likely score lower.
- **Sample, not full data.** Results are for a 20% sample over 26 weeks. About 18% of scored customers had no history in that window, even though some of them had shopped with H&M before.
- **Offline only.** MAP@12 on past weeks doesn't prove more sales. That needs a live A/B test, which I've designed but not run.
- **Small search test.** 16 requests, with rules I wrote myself. All answers came from the lighter Gemini model, because the main one was overloaded during the run.
- **Prices are scaled by H&M**, so nothing here says anything about real revenue.

## Fairness
I didn't use any protected attributes. The only personal detail available is age. I tested it for new customers, but the gain wasn't statistically clear, so the final system doesn't use it for them.

