import numpy as np
import pandas as pd
import random
from orderbook import OrderBook
from bintrees import RBTree
import plotly.graph_objs as go
import scipy.stats as st

#%%

spread_result = pd.DataFrame(columns = ['Simulation_ID', 'Mean','StdDev','ConfInterval'])
idle_time_result = pd.DataFrame(columns = ['Simulation_ID', 'Mean','StdDev','ConfInterval'])
Numb_Sim = 100

for s in range (0, Numb_Sim):

    initial_price = 20.0
    pip = 0.01
    interval_range = 40
    lot_size = 100

    points = np.array(range(int(initial_price / pip - interval_range), 
                            int(initial_price / pip + interval_range))) / int(1 / pip) # + 1

    df = pd.DataFrame(points, columns = ['price'])

    df['mu'] = 1
    df['lambda'] = 0.2
    df['volume'] = 5
    df['gamma'] = 0.1

    # df['mu'] = 1
    # df['lambda'] = 1 * df['mu']
    # df['volume'] = 5
    # df['gamma'] = 9 * df['mu']

    # df = pd.read_csv('petr4_min.csv')
    # initial_price = 21.39
    # points = np.array(df['price'])
    # df['mu'] = 0.5 + df['mu'] / 5
    # df['volume'] = 5 #df['volume'].mean() / 480
    # df['lambda'] = 0.1 + df['mu'] / 10
    # df['gamma'] = 0.15

    #%%
    lob = OrderBook() # Create a LOB object
    max_iterations = 200
    start_at = 20
    last_price = initial_price

    bid, ask = initial_price - pip, initial_price
    mid = (bid + ask) / 2
    order_id = 0
    limit_orders = RBTree()
    total_traded = dict()
    time = 0

    price_history = []
    bid_history = []
    ask_history = []
    volume_history = []
    book_bid_volume = []
    book_ask_volume = []
    idle_time_history = []

    def add_trades(total_traded, trades):
        for trade in trades:
            total_traded.setdefault(trade['price'], {'qty': 0, 'orders': 0})
            total_traded[trade['price']]['qty'] += trade['qty']
            total_traded[trade['price']]['orders'] += 1
        return total_traded

    def get_last_price(last_price, trades):
        return trades[len(trades) - 1]['price'] if len(trades) > 0 else last_price

    def get_traded_volume(trades):
        return sum([trade['qty'] for trade in trades] + [0])

    # Loop
    for t in range(max_iterations):
        traded_volume = 0
        next_orders = []
        
        # Limit order cancellations
        
        for idNum, order in limit_orders.items():
            if t >= order['time_limit']:
                lob.cancelOrder(order['side'], idNum)
                limit_orders.remove(idNum)

        # Limit order arrivals
        
        df['orders'] = df['lambda'].apply(lambda x: np.random.poisson(x))
        for idx, row in df.iterrows():
            for i in range(int(row['orders'])):
                
                if row['price'] < ask:
                    side = 'bid'
                    order_id = order_id + 1
                    order = {
                        'type' : 'limit', 
                        'side' : side, 
                        'qty' : np.random.poisson(row['volume']) * lot_size, 
                        'price' : row['price'],
                        'tid' : order_id
                        }
                    if order['qty'] > 0:
                        next_orders.append(order)
                        
                if row['price'] > bid:
                    side = 'ask'
                    order_id = order_id + 1
                    order = {
                        'type' : 'limit', 
                        'side' : side, 
                        'qty' : np.random.poisson(row['volume']) * lot_size, 
                        'price' : row['price'],
                        'tid' : order_id
                        }
                    if order['qty'] > 0:
                        next_orders.append(order)
        
        # Market order arrivals
        
        if t >= start_at:
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
                    next_orders.append(order)
                
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
                    next_orders.append(order)

        # Process orders

        random.shuffle(next_orders)
        for order in next_orders:
            trades, order = lob.processOrder(order, False, False)
            if order:
                order['time_limit'] = t + np.random.geometric(row['gamma'])
                limit_orders.insert(order['idNum'], order)
            if trades:
                idle_time_history.insert(len(idle_time_history), time)
                time = 0
            else:
                time+=1
            total_traded = add_trades(total_traded, trades)
            last_price = get_last_price(last_price, trades)
            traded_volume += get_traded_volume(trades)      
      
            # Next prices
                    
            bid = lob.getBestBid() if str(lob.getBestBid()) != 'None' else bid
            ask = lob.getBestAsk() if str(lob.getBestAsk()) != 'None' else ask
            mid = (bid + ask) / 2
            
            price_history.append(last_price)
            bid_history.append(bid)
            ask_history.append(ask)
            # volume_history.append(traded_volume)
        
            # x_bid, x_ask, y_bid, y_ask = [], [], [], []
            # for price in points:
            #     if price < mid:
            #         x_bid.append(price)
            #         y_bid.append(lob.getVolumeAtPrice('bid', price))
            #     elif price > mid:
            #         x_ask.append(price)
            #         y_ask.append(lob.getVolumeAtPrice('ask', price))
            # book_bid_volume.append(sum(y_bid))
            # book_ask_volume.append(sum(y_ask))
            
        # Plot chart

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
        
        # print('Time', t)
        # print('Total Volume Bid', sum(y_bid))
        # print('Total Volume Ask', sum(y_ask))
        
    # print(lob)

    # Simulation Evolution
        
    # traces = [
    #     go.Scatter(x = list(range(len(price_history))), y = price_history, name = 'Last', marker_color = 'green'),
    #     go.Scatter(x = list(range(len(price_history))), y = bid_history, name = 'Bid', marker_color = 'blue'),
    #     go.Scatter(x = list(range(len(price_history))), y = ask_history, name = 'Ask', marker_color = 'red'),
    #     go.Bar(x = list(range(len(price_history))), y = volume_history, name = 'Volume', marker_color = 'rgba(0, 0, 0, 0.5)', yaxis = 'y2')
    #     ]
    # layout = {
    #     'title': 'Evolução da Simulação',
    #     'separators': '.',
    #     'yaxis': dict(tickformat = '.2f', gridcolor = 'grey', title = 'Preço', domain = [0.3, 1]),
    #     'yaxis2': dict(
    #         title = 'Volume',
    #         side = 'right',
    #         domain = [0, 0.2],
    #         anchor = 'x'
    #     ),
    #     'margin': dict(l = 40, r = 40, b = 40, t = 60, pad = 20),
    #     'plot_bgcolor': 'rgb(255, 255, 255)',
    #     }
    # fig = go.Figure(data = traces, layout = layout)
    # fig.show()

    # Spread
    spread_array = np.array(ask_history) - np.array(bid_history)
    print('Spread Mean', spread_array.mean())
    print('Spread StdDev', spread_array.std())
    print('Spread ConfInterval', st.t.interval(0.95, len(spread_array)-1, loc=np.mean(spread_array), scale=st.sem(spread_array)))

    spread_result = spread_result.append({
                                            'Simulation_ID': s+1,
                                            'Mean': spread_array.mean(),
                                            'StdDev': spread_array.std(),
                                            'ConfInterval': st.t.interval(0.95, len(spread_array)-1, loc=np.mean(spread_array), scale=st.sem(spread_array))[1]-st.t.interval(0.95, len(spread_array)-1, loc=np.mean(spread_array), scale=st.sem(spread_array))[0]
                                        }, ignore_index=True)

    # Idle_time
    idle_time_array = np.array(idle_time_history)
    print('Spread Mean', idle_time_array.mean())
    print('Spread StdDev', idle_time_array.std())
    print('Spread ConfInterval', st.t.interval(0.95, len(idle_time_array)-1, loc=np.mean(idle_time_array), scale=st.sem(idle_time_array)))

    print(idle_time_history)

    idle_time_result = idle_time_result.append({
                                                    'Simulation_ID': s+1,
                                                    'Mean': idle_time_array.mean(),
                                                    'StdDev': idle_time_array.std(),
                                                    'ConfInterval': st.t.interval(0.95, len(idle_time_array)-1, loc=np.mean(idle_time_array), scale=st.sem(idle_time_array))[1]-st.t.interval(0.95, len(idle_time_array)-1, loc=np.mean(idle_time_array), scale=st.sem(idle_time_array))[0]
                                                }, ignore_index=True)

    # Book Evolution
    
    # traces = [
    #     go.Scatter(x = list(range(len(price_history))), y = np.array(book_bid_volume) + np.array(book_ask_volume), name = 'Total', marker_color = 'green'),
    #     go.Scatter(x = list(range(len(price_history))), y = book_bid_volume, name = 'Bid', marker_color = 'blue'),
    #     go.Scatter(x = list(range(len(price_history))), y = book_ask_volume, name = 'Ask', marker_color = 'red'),
    #     ]
    # layout = {
    #     'title': 'Evolução do Book',
    #     'separators': '.',
    #     'yaxis': dict(gridcolor = 'grey', title = 'Volume'),
    #     'margin': dict(l = 40, r = 40, b = 40, t = 60, pad = 20),
    #     'plot_bgcolor': 'rgb(255, 255, 255)',
    #     }
    # fig = go.Figure(data = traces, layout = layout)
    # fig.show()

    # # Book Volume
    # print('Book Volume', (np.array(book_bid_volume) + np.array(book_ask_volume)).mean())

    # Volume at Price

    # volumes = pd.DataFrame(total_traded).T
    # traces = [
    #     go.Bar(x = volumes['qty'], y = volumes.index, name = 'Volume', marker_color = 'green', orientation = 'h')
    #     ]
    # layout = {
    #     'title': 'Volume por Preço',
    #     'separators': '.',
    #     'xaxis': dict(title = 'Quantidade', gridcolor = 'grey'),
    #     'yaxis': dict(tickformat = '.2f', nticks = 10, title = 'Preço'),
    #     'margin': dict(l = 40, r = 40, b = 40, t = 60, pad = 20),
    #     'plot_bgcolor': 'rgb(255, 255, 255)',
    #     }
    # fig = go.Figure(data = traces, layout = layout)
    # fig.show()

    # # Traded Volume
    # print('Traded Volume', sum(volumes['qty']))

writer = pd.ExcelWriter('result.xlsx')
spread_result.to_excel(writer,'spread', index=False)
idle_time_result.to_excel(writer,'idle_time', index=False)
writer.save()