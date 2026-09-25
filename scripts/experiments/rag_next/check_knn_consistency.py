import components as C
from common import macro_f1
uids = C.query_uids("dev"); y = C.labels_of(uids)
for v in ["q3L36", "q7L28", "q14L48"]:
    print(v, "components.knn_vote PS@9 dev F1 = %.4f" % macro_f1(y, C.knn_vote("dev", "PS", v, 9).argmax(1)))
