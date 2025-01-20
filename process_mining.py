#!/usr/bin/env python

### Display
from IPython.display import display
## Data Handling
import pandas as pd
import pm4py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# please do not change or delete this cell
random.seed(42)
np.random.seed(42)

log = pm4py.read_xes('process_mining/event_log.xes')


# (a) ACTIVITIES

value_cnts_df = log['concept:name'].value_counts().to_frame()

ax = sns.barplot(data = value_cnts_df, x="concept:name", y="count")
ax.set_title("Activity frequencies")
ax.set_xlabel("activity name")
ax.set_ylabel("absolute frequency")
plt.show()

# number of events
display(f'Number of events: {len(log)}')


# (b) LOOK AT THE CASES

cases_num = len(log['case:concept:name'].unique())
offers_num = len(log['OfferID'].unique())
display(f'Number of cases: {cases_num}; Number of offers: {offers_num}; Mean of offers per case: {(offers_num/cases_num):.3f}')


# (c) PLOT START/END ACTIVITIES FREQUENCY

dfg, start_activities, end_activities = pm4py.discover_directly_follows_graph(log)
pm4py.view_dfg(dfg=dfg, start_activities=start_activities, end_activities=end_activities)

data = []
for k in iter(start_activities):
	data.append([k, start_activities[k], "start"])

for k in iter(end_activities):
	data.append([k, end_activities[k], "end"])

df = pd.DataFrame(data=data, columns=["activity_name", "frequency", "activity_type"])

ax = sns.barplot(data = df, x="activity_name", y="frequency", hue="activity_type")
ax.set_title("Start/End activity frequencies")
ax.set_xlabel("activity name")
ax.set_ylabel("absolute frequency")
plt.show()



# (d) PETRI NET MODEL WITH INDUCTIVE MINER

proc_tree = pm4py.discover_process_tree_inductive(log)
pm4py.view_process_tree(proc_tree)

net, im, fm = pm4py.discover_petri_net_inductive(log)
pm4py.view_petri_net(petri_net=net, initial_marking=im, final_marking=fm)
pm4py.save_vis_petri_net(petri_net=net, initial_marking=im, final_marking=fm, file_path="petri_net.svg")

# (e) UNDERSTAND THE PROCESS

# Activity sequence: Create Offer, Created, Sent (mail and online), Returned, Accepted
# The load application is accepted, and then the offer is created and an email is sent to the customer. The customer returns to the finance organization and accepts the offer.


# (f) FIND NONSENSE ACTIVITY SEQUENCE

# Activity sequence: Create Offer, Created, Sent (online only)
# Activity sequence: Create Offer, Created, Sent (online only), Create Offer, Created, Sent (online only)
# Activity sequence: Create Offer, Created, Sent (online only), Create Offer, Created, Sent (online only), Create Offer, Created, Sent (online only)
# One of the above sequences must not exist. Since there are only two cases with the end activity of "Sent (online only)". It doesn't make sense to make many offers before getting any response from the customer. The Inductive Miner discover loops, but it doesn't specify how many times the loop might repeat.


# (g) CHART SHOWING PERCENTAGE OF CASES

variants = pm4py.get_variants_as_tuples(log)
variants_sorted = sorted(variants.items(), key=lambda x: x[1], reverse=True)

t = [0]
xsum = 0
t += [xsum := xsum + x[1] for x in variants_sorted]
d = {}
for i in range(len(t)):
	d[i] = (t[i]/cases_num) * 100


variants_frequencies = pd.Series(d)
variants_frequencies.plot(x="number of variants", y="accumulated of sum of the percentage of cases covered")
plt.show()


# (h)

display(f"Fewer than 10: {sum([variants[x] < 10 for x in variants])}")

# The chart shows that we need at least 12 variants to cover 85% of cases.
# 60% of cases are covered by the two most frequent variants.
# There are 427 variants representing fewer than 10 cases.

# (i)

filtered_log = pm4py.filter_variants(log, variants=[ x[0] for x in variants_sorted[:5]])
net, im, fm = pm4py.discover_petri_net_inductive(filtered_log)
pm4py.view_petri_net(petri_net=net, initial_marking=im, final_marking=fm)

fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
display(f"log_fitness:{fitness['log_fitness']:.3f}")

# (j) HISTOGRAM FOR INDIVIDUAL TRACE FITNESS VALUES

bins = np.arange(0, 1.02, 0.02)
individuals = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)

hist = sns.histplot([x['trace_fitness'] for x in individuals], bins=bins)
hist.set_title("individual trace fitness values distribution")
hist.set_xlabel("fitness value")
hist.set_ylabel("count")
plt.show()

# (k) EXPLANATION OF TWO PHENOMENA

# (1)
# (2)

# (l) A BAR CHART SHOWING THE MEAN THROUGHPUT TIME


mean_days = []
variant_casenums = []
edges = []
edge = 0
day_secs = 60 * 60 * 24
df1_data = []

variant_index = 0
for x in variants_sorted[:35]:
	filtered_log = pm4py.filter_variants(log, [x[0]])
	a=pm4py.get_all_case_durations(filtered_log)
	offers = sum([i == "Create Offer" for i in x[0]])
	variant_mean_throughput_days = sum(a)/len(a)/day_secs
	df1_data.append({"throughput time": variant_mean_throughput_days, "offers": offers})
	mean_days.append(variant_mean_throughput_days)
	variant_casenums.append(len(a))
	edges.append(edge)
	edge += len(a)
	variant_index += 1

bar = plt.bar(edges, mean_days, variant_casenums, align='edge', edgecolor='black')
# bar.set_title("The mean throughput time (in days) of the 35 most frequent variants and the number of cases covered by each variant")
plt.show()

# (m)

df1 = pd.DataFrame(data = df1_data)
df1_barplot = sns.barplot(data=df1, x="offers", y="throughput time")
df1_barplot.set_title("Number of offers created impact on the throughput time")
plt.show()

# The plot shows an indication that the more offers presented to a customer, the more throughput time it takes.