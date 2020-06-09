import numpy as np
import pandas as pd
import random
from orderbook import OrderBook
from bintrees import RBTree
import plotly.graph_objs as go
import scipy.stats as st
import time

#%%
start_time = time.time()

spread_result = pd.DataFrame(columns = ['Simulation_ID', 'Mean','StdDev','ConfInterval'])
efficiency_result = pd.DataFrame(columns = ['Simulation_ID', 'Mean','StdDev','ConfInterval'])
lifetime_result = pd.DataFrame(columns = ['Simulation_ID', 'Mean','StdDev','ConfInterval'])

numb_sim = 1
max_iterations = 200
start_at = 20
initial_price = 20.0
pip = 0.01
interval_range = 40
lot_size = 100

for s in range (0, numb_sim):

    points = np.array(range(int(initial_price / pip - interval_range), 
                            int(initial_price / pip + interval_range))) / int(1 / pip) # + 1

    df = pd.DataFrame(points, columns = ['price'])

    df['mu'] = 1
    df['lambda'] = 0.2
    df['volume'] = 5
    df['gamma'] = 0.1

    lob = OrderBook() # Create a LOB object
    last_price = initial_price

    bid, ask = initial_price - pip, initial_price
    mid = (bid + ask) / 2
    order_id = 0
    limit_orders = RBTree()
    total_traded = dict()
    total_trades = 0
    total_orders = 0
    
    bid_history = []
    ask_history = []
    born_and_dead_history = dict()

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
            total_orders += 1

            if order:
                # order was created or not fully executed
                idNum = order['idNum']
                order['time_limit'] = t + np.random.geometric(float(df[df['price'] == order['price']]['gamma']))
                limit_orders.insert(idNum, order)
                born_and_dead_history[idNum] = {
                                                    'idNum': idNum,
                                                    'born': t,
                                                    'dead': order['time_limit']
                                                }

            if trades:
                for trade in trades:
                    idNum = trade['party1'][2] # trade['party1'] = [tid, side, idNum]
                    born_and_dead_history[idNum]['dead'] = t
                total_trades += 1
      
            # Next prices
            bid = lob.getBestBid() if str(lob.getBestBid()) != 'None' else bid
            ask = lob.getBestAsk() if str(lob.getBestAsk()) != 'None' else ask
            mid = (bid + ask) / 2
            
            bid_history.append(bid)
            ask_history.append(ask)

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

    # Efficiency
    efficiency = total_trades/total_orders
    print('Efficiency', efficiency)
    efficiency_result = efficiency_result.append({
                                                    'Simulation_ID': s+1,
                                                    'Mean': efficiency,
                                                    'StdDev': 0,
                                                    'ConfInterval': 0,
                                                }, ignore_index=True)

    # Lifetime
    bd = pd.DataFrame.from_dict(born_and_dead_history, orient='index')
    bd['lifetime'] = bd['dead'] - bd['born']
    # bd.to_excel('born_and_dead_history_' + str(s) + '.xlsx')
    lifetime_array = np.array(bd['lifetime'])
    print('Lifetime Mean', lifetime_array.mean())
    print('Lifetime StdDev', lifetime_array.std())
    print('Lifetime ConfInterval', st.t.interval(0.95, len(lifetime_array)-1, loc=np.mean(lifetime_array), scale=st.sem(lifetime_array)))

    lifetime_result = lifetime_result.append({
                                            'Simulation_ID': s+1,
                                            'Mean': lifetime_array.mean(),
                                            'StdDev': lifetime_array.std(),
                                            'ConfInterval': st.t.interval(0.95, len(lifetime_array)-1, loc=np.mean(lifetime_array), scale=st.sem(lifetime_array))[1]-st.t.interval(0.95, len(lifetime_array)-1, loc=np.mean(lifetime_array), scale=st.sem(lifetime_array))[0]
                                        }, ignore_index=True)

    print(s+1, 'of', numb_sim)

writer = pd.ExcelWriter('result.xlsx')
spread_result.to_excel(writer,'spread', index=False)
efficiency_result.to_excel(writer,'efficiency', index=False)
lifetime_result.to_excel(writer,'lifetime', index=False)
writer.save()

str_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
print(f"Finished. Elapsed time: {str_time}")
