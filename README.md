# USL Player Analysis
## 1. Executive Summary
### Problem
Oakland Roots needs help recruiting talented soccer players to their team. In particular, they are looking to find players with skillsets lacking in the Roots team and at an affordable price. 
### Solution
With data sourced and scraped from soccer statistics websites, we have provided a comprehensive analysis/plan of attack for Oakland Roots. We delve into player statistics, team dynamics, field layouts, and feature/skillset importance to make better-informed recruitment decisions.
### Highlights 
Using various analytic techniques, we uncovered resources & statistics that provide the Oakland Roots team strategies to recruit talent. These include 3 main categories 
- Talent-based recruitments (based on specific features of desired players)
- Similarity-based recruitment (comparing to other players)
- Team-based recommendations & blindspots (based on what team dynamics lead to winning championships).
- More specifically, this breaks down into these loosely-grouped analytic categories:
    - Undervalued Players (Market Value Model)
    - Similar Players (Cosine Similarity)
    - mportant Features of Valuable Players (Random Forest)
    - Important Features Per Position
    - Team-by-Team Comparisons
    - Statistics by Features
    - Player Country of Origin (Choropleth)
    - Radar Plots Comparing Most Similar Players
## 2. Data Retrieval
### Databases 
#### Wyscout (2020 & 2021) 
The Wyscout dataset is retrieved from the Wyscout data platform as provided by the client. Wyscout is the largest soccer data database on the internet. It compares players in the USL League One — ranking players by shots, crosses, & successful tackles in the USL Championship Games. 
#### American Soccer Analysis Dataset
The American Soccer Analysis (ASA) dataset is downloaded to visualize the data. The dataset contains three separate datasets of xGoals, xPass, and Goals Added (g+). The metric xGoals (xG) assesses the probability (%) of any shot scoring (a goal). Specifically, it quantifies the difficulty of a scoring shot using various predictive metrics — players' goals, key passes, & assists. Likewise, the metric xPass (xP) assesses the probability (%) of a pass being successful (making it to a teammate).
### Webscraping
#### TransferMarkt
Transfermarkt is a website where anyone can look up a soccer team with player statistics, game stats, & a player's market value ($). 
#### Sofa Score
The SofaScore ratings for 2020 and 2021 were scraped from the site. This dataset was merged with Wyscout to construct feature selection, ratings, and a salary model.
## 2. Data Processing
### Position Categorization
Owing to the complexity of the datasets in this project, biases afflict the descriptions of players' positions from different sources. Moreover, stats on positions will also differ due to the opportunities different positions are presented with. Offensive players, for instance, will have more goal-scoring opportunities whilst defensive players rank higher in deep passes. So — categorizing by position is inherently valuable. Nonetheless, it makes sense to aggregate players in similar positions so that the categories are more meaningfully predictive when we conduct machine learning.
### Data Consolidation
In order to make the data better serve the subsequent progress, we decided to merge the Rating column of the data from the Wyscout source with the data from the SofaScore source via the Player column. In order to make the data better serve the subsequent progress, we decided to merge the Rating column of the data from the Wyscout source with the data from the SofaScore source via the Player column. However, since the player names of the SofaScore source data are fully spelled, they do not match the Wyscout data. We match by constructing a regular expression that keeps the player's last name and replaces the player's first name with an acronym. Player names with middle names or special characters that are not alphabetic are matched using manual substitution.
## 3. EDA
Using Python package ProfileReport to generate EDA report.

```
data_Transfer = pd.read_csv('datasets/new/data_TansferMarkt_2021_new.csv')
Transfermarkt_Data_profile = pf.ProfileReport(data_Transfer, title = "Transfermarkt Data Report 2021")
Transfermarkt_Data_profile.to_widgets()
Transfermarkt_Data_profile.to_file('Transfermarkt Data Report 2021.html')
```

## 4. Data visualization (Parts)
Plotting the age of soccer players reveals that more than half the players are younger than 25, peaking at 24-year-olds. 18 features a peak since recruitment efforts ramp up once the player enters adulthood. After 25, however, there is a sharp decline. Some skilled players play well into their 30s, while most end their soccer careers early. This is likely more indicative of career & lifestyle options available to such players than the decrepit age. However, past the mid-30s it becomes impossible to maintain peak physical condition.

![image](https://github.com/jdenggao/USL-Player-Analysis/assets/112433825/aaf5ff9c-b553-4f94-b7b1-40aea3d5f0a5)

By mapping the number of players and market value of different teams with data from Wyscout sources. We can find that North Carolina FC and Toronto II are the two teams with the most players, with 31 players. Although the number of players in the same, we can find that the total value of Toronto II's players is more than four times the total value of North Carolina FC's players by the market value line chart. North Texas and New England II are in the second tier, followed by Toronto II, which is the most powerful and well-financed of all teams. The remaining teams are in the third tier, and the Chattanooga Red Wolves have the lowest stats in comparison.

![image](https://github.com/jdenggao/USL-Player-Analysis/assets/112433825/ee3f3330-99b3-4708-9b6c-7ad4ee70f4a8)

## 5. Features selection
In our models, we took the counsel of our advisor as to what features mattered most by position. We compiled it into the table below. Given these groupings, we used each feature’s perceived importance as weighted values for predictive models of skill set. 
<br>
To be able to further select the important features from the features of each position above, we decided to build a random forest model with the features of each position above as the independent variable and 'Rating' as the dependent variable.
```
  forest = RandomForestClassifier(oob_score=True, n_estimators=100, random_state=100, n_jobs=-1)
  forest.fit(x_train, y_train)
  
  # select features which threshold larger then 0.025
  selector = SelectFromModel(forest, threshold=0.025)
  features_important = selector.fit_transform(x_train, y_train)
  model = forest.fit(features_important, y_train)
  selected_features = model.feature_importances_
```
Following two plots are for the feature selection on the Fullback position. Fullbacks are responsible for defending both sides of the field. Their task is to prevent the opponent from passing the ball from the sideline into the penalty area, and they also help make passes for attacks. In the scatter plot, the important features of a FB are key passes per 90, duels per 90, and shots blocked per 90. These are all essential features for FB to prevent opponents from passing the ball, creating a chance for goals. There are also box and violin plots of the features to show the min, median, max, 0.25, 0.75, upper max, and outliers for each of the features.

![image](https://github.com/jdenggao/USL-Player-Analysis/assets/112433825/1336dc8e-ad65-48a3-9d74-99a12b59c88c)

![image](https://github.com/jdenggao/USL-Player-Analysis/assets/112433825/c141fbd4-1b3a-4386-8ccb-fd38a1ea38f5)

## 6. Cosine distance
The client provided us with four ideal players (Reggie Cannon, James Sands, Ronaldo Damus, and Cole Bassett). When planning a player acquisition, these players have the statistics the client thought would best match his preference for these four positions (LB/RB, CB, FWD, and AM). These four positions were categorized into broader categories in previous steps (CB, CF, and FB). To locate other players similar to these five players, we calculated the cosine distance between the ideal players and similar players.

<br>
By extracting the model player's statistics, and the essential features from the positions data generated by the RF model. The dataset was normalized and separated the data into x, all the players' features, and y, the player's name. Next, we calculated the cosine similarity between the players. The angle between the players will give the percent similarity between the ideal player and comparable players. We then use cosine distance (1-cosine similarity) to determine the distance between the two players. If the players' cosine distance is closer to 0, the two players are alike. Players with more significant cosine similarity and the least cosine distance are the top similar players.

<br>

```
def Cosine_Distance(Player_data, Position_data):
    
    data_copy = Position_data.copy()
    # select player
    name = data_Sample[data_Sample["Player"] == Player_data]
    # select features in data_sample dataset
    feature = name[[i for i in data_copy.columns.tolist()]]
    
    # normalize
    sclar = preprocessing.StandardScaler()
    data_Norm = pd.DataFrame(sclar.fit_transform(pd.concat([feature, data_copy])))
    
    x = data_Norm.iloc[0:1]
    y = data_Norm.iloc[1:]
    
    cos_s = []
    cos_d = []
    player = []
   
    # cos_sim & cos_dis
    for i in range(len(y)):
        simi = cosine_similarity(x, y[i:i+1]).tolist()[0][0]
        cos_s.append(simi)
        dist = paired_distances(x, y[i:i+1], metric='cosine').tolist()[0]
        cos_d.append(dist)
    
    for i in Position_data.index:
        player.append(data_Wy['Player'][i])
        
    data0 = data_copy
    data0.loc[:, 'Cosine Similarity'] = cos_s
    data0.loc[:, 'Cosine Distances'] = cos_d
    data0.loc[:, 'Player'] = player
    data0 = data0.sort_values(by=['Cosine Similarity'],axis=0,ascending=[False]) 
    
    return data0
```

<br>

To enable our clients to compare all players, we have implemented this feature in Dash. The client can select the position of the player he wants to search for and then select the two players he wants to compare to get an idea of the player. This Dashboard can be viewed in a Jupyter notebook.

```
def Polar(Player_data, Selected_Player_data):
    
    data_copy = Selected_Player_data.copy().iloc[0:1,:-3]

    # select player
    name = data_Sample[data_Sample["Player"] == Player_data]
    # select features in data_sample dataset
    feature = name[[i for i in data_copy.columns.tolist()]]
    
    data = pd.concat([feature, data_copy])
    
    fig = make_subplots(rows=1, cols=2,specs=[[{"type": "Polar"},{"type": "Polar"}]])

    R1=[]
    theta1=[]
    R2=[]
    theta2=[]
    R3=[]
    theta3=[]
    R4=[]
    theta4=[]
    
    for i in data.columns[1:]:
        
        if max(data[i]) < 20:
            R1.append(feature[i].iloc[0])
            theta1.append(i)
            R2.append(data_copy[i].iloc[0])
            theta2.append(i)
        else:   
            R3.append(feature[i].iloc[0])
            theta3.append(i)
            R4.append(data_copy[i].iloc[0])
            theta4.append(i)
            
    fig.add_trace(go.Scatterpolar(
          r = R1,
          theta = theta1,
          fill = 'toself',
          marker_color='rgb(47,138,196,30)',
          name = Player_data,
          ),row=1, col=1)
    
    fig.add_trace(go.Scatterpolar(
          r = R2,
          theta = theta2,
          fill = 'toself',
          marker_color='rgb(229,210,245,0.2)',
          name = Selected_Player_data.Player.values[0]),row=1, col=1)
      
    fig.add_trace(go.Scatterpolar(
          r = R3,
          theta = theta3,
          fill = 'toself',
          marker_color='rgb(47,138,196,30)',
          name = Player_data),row=1, col=2)
    
    fig.add_trace(go.Scatterpolar(
          r = R4,
          theta = theta4,
          fill = 'toself',
          marker_color='rgb(229,210,245,0.2)',
          name = Selected_Player_data.Player.values[0]),row=1, col=2)

    fig.layout.update(
        go.Layout(
        polar = dict(
            radialaxis = dict(
                visible = True,)),
        showlegend = True,
        title = "{} vs {}".format(Player_data, Selected_Player_data.Player.values[0]),
        height=400, width=1300,
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        ))
    
    return py.offline.iplot(fig)
```

![image](https://github.com/jdenggao/USL-Player-Analysis/assets/112433825/1df039a7-6303-405f-8c4a-23022d3ad3cd)

## 7. Rating model & Market value model
By obtaining the rating of the players' performance characteristics from the SofaScore website and merging it with the Wy data set using data processing techniques, a complete data set was generated for analysis and modeling. The rating score is the target value(y), and each position's different essential performance metrics are used as input variables for modeling and prediction. The rating prediction model is built using the player performance data of the 2020 season combined with the random forest algorithm, and the player performance data of the 2021 season is input to the model as the independent variable required for prediction. The rating value of the 2021 season is predicted and fetched through the model. At the same time, we compare the predicted Ratings with the real values. We considered the players with a low real Rating value and high predicted Rating value as potential players. This group of players is considered the players who can reveal their potential through strenuous practical training to perform better in the future. Then, all models screen the candidates and obtain the player's personal information (e.g., left and right foot habits, contract information, salary information, etc.)

<br>

We modeled and predicted by using salary data from Wy dataset and different characteristics of different positions. We use the important data of players in 2020 season combined with Random Forest algorithm to build the salary prediction model, and input the performance data of players in 2021 season as the variables needed for prediction into the model, and get the salary value of players in 2021 season through the model. At the same time, we compare the projected salary value with the real salary value, and we define the players whose real salary value is lower than the projected salary value as valuable players with potential, and use them as the player selection evaluation criteria. Through the random forest model, we exported the top 5 players of each position market value and stored the data in 'datasets/results/Top5_Market_value_2021_' + p + '.csv' in order by position

## Detail Report 
[Capstone Project Final Report Root Team.pdf](https://github.com/jdenggao/USL-Player-Analysis/files/12600559/Capstone.Project.Final.Report.Root.Team.pdf)
