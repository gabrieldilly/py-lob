import numpy as np
import pandas as pd
from orderbook import OrderBook
from bintrees import RBTree
import plotly.graph_objs as go

#%%

initial_price = 20.0
pip = 0.01
interval_range = 40
lot_size = 100

points = np.array(range(int(initial_price / pip - interval_range), 
                        int(initial_price / pip + interval_range))) / int(1 / pip) # + 1

df = pd.DataFrame(points, columns = ['price'])

df['mu'] = 1
df['lambda'] = 0.1
df['volume'] = 5
df['gamma'] = 0.1

# df['mu'] = 1
# df['lambda'] = 1 * df['mu']
# df['volume'] = 5
# df['gamma'] = 9 * df['mu']


#%%
lob = OrderBook() # Create a LOB object
max_iterations = 100
start_at = 10
last_price = initial_price
price_history = []
bid_history = []
ask_history = []
bid, ask = initial_price - pip, initial_price
mid = (bid + ask) / 2
order_id = 0
limit_orders = RBTree()
total_traded = dict()

def add_trades(total_traded, trades):
    for trade in trades:
        total_traded.setdefault(trade['price'], {'qty': 0, 'orders': 0})
        total_traded[trade['price']]['qty'] += trade['qty']
        total_traded[trade['price']]['orders'] += 1
    return total_traded

def get_last_price(last_price, trades):
    return trades[len(trades) - 1]['price'] if len(trades) > 0 else last_price

# Loop
for t in range(max_iterations):
    
    # Limit order cancellations
    
    for idNum, order in limit_orders.items():
        if t > order['time_limit']:
            lob.cancelOrder(order['side'], idNum)
            limit_orders.remove(idNum)

    # Limit order arrivals
    
    df['orders'] = df['lambda'].apply(lambda x: np.random.poisson(x))
    for idx, row in df.iterrows():
        side = 'bid' if row['price'] < mid else 'ask'
        for i in range(int(row['orders'])):
            order_id = order_id + 1
            order = {
                'type' : 'limit', 
                'side' : side, 
                'qty' : np.random.poisson(row['volume']) * lot_size, 
                'price' : row['price'],
                'tid' : order_id
                }
            if order['qty'] > 0:
                trades, order = lob.processOrder(order, False, False)
                order['time_limit'] = t + np.random.exponential(1 / row['gamma'])
                limit_orders.insert(order['idNum'], order)
                total_traded = add_trades(total_traded, trades)
                last_price = get_last_price(last_price, trades)
    
    
    # Market order arrivals
      
    if t > start_at:
        mkt_buy = df[df['price'] == bid].to_dict(orient = 'records')[0]
        for i in range(int(np.random.poisson(mkt_buy['mu']))):
            order_id = order_id + 1
            order = {
                'type' : 'market', 
                'side' : 'bid', 
                'qty' : np.random.poisson(mkt_buy['volume']) * lot_size, 
                'tid' : order_id
                }
            if order['qty'] > 0:
                trades, id_num = lob.processOrder(order, False, False)
                total_traded = add_trades(total_traded, trades)
                last_price = get_last_price(last_price, trades)
            
        mkt_sell = df[df['price'] == ask].to_dict(orient = 'records')[0]
        for i in range(int(np.random.poisson(mkt_sell['mu']))):
            order_id = order_id + 1
            order = {
                'type' : 'market', 
                'side' : 'ask', 
                'qty' : np.random.poisson(mkt_sell['volume']) * lot_size, 
                'tid' : order_id
                }
            if order['qty'] > 0:
                trades, id_num = lob.processOrder(order, False, False)
                total_traded = add_trades(total_traded, trades)
                last_price = get_last_price(last_price, trades)
    
    # Next prices
            
    bid = lob.getBestBid() if str(lob.getBestBid()) != 'None' else bid
    ask = lob.getBestAsk() if str(lob.getBestAsk()) != 'None' else ask
    mid = (bid + ask) / 2
    
    price_history.append(last_price)
    bid_history.append(bid)
    ask_history.append(ask)
    
    # Plot chart
    
    # x_bid, x_ask, y_bid, y_ask = [], [], [], []
    # for price in points:
    #     if price < mid:
    #         x_bid.append(price)
    #         y_bid.append(lob.getVolumeAtPrice('bid', price))
    #     elif price > mid:
    #         x_ask.append(price)
    #         y_ask.append(lob.getVolumeAtPrice('ask', price))
        
    # traces = [
    #     go.Bar(x = x_bid, y = y_bid, name = 'Bid', marker_color = 'blue'),
    #     go.Bar(x = x_ask, y = y_ask, name = 'Ask', marker_color = 'red')
    #     ]
    # layout = {
    #     'title': 'Book de Ofertas',
    #     'separators': '.',
    #     'xaxis': dict(tickformat = '.2f', nticks = 10),
    #     'yaxis': dict(gridcolor = 'grey'),
    #     'margin': dict(l = 40, r = 40, b = 40, t = 60, pad = 20),
    #     'plot_bgcolor': 'rgb(255, 255, 255)',
        
    #     }
    # fig = go.Figure(data = traces, layout = layout)
    # fig.show()
    

# print(lob)
    
traces = [
    go.Scatter(x = list(range(max_iterations)), y = price_history, name = 'Last', marker_color = 'green'),
    go.Scatter(x = list(range(max_iterations)), y = bid_history, name = 'Bid', marker_color = 'blue'),
    go.Scatter(x = list(range(max_iterations)), y = ask_history, name = 'Ask', marker_color = 'red')
    ]
layout = {
    'title': 'Evolução do Preço',
    'separators': '.',
    'yaxis': dict(tickformat = '.2f', gridcolor = 'grey'),
    'margin': dict(l = 40, r = 40, b = 40, t = 60, pad = 20),
    'plot_bgcolor': 'rgb(255, 255, 255)',
    }
fig = go.Figure(data = traces, layout = layout)
fig.show()
