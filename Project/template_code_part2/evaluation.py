import math

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        relevant_docs = set(true_doc_IDs)
        retrieved_docs = query_doc_IDs_ordered[:k]
        num_relevant_retrieved = len(relevant_docs.intersection(retrieved_docs))
        precision = num_relevant_retrieved / k if k > 0 else 0
        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        total_precision = sum(self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrels[i]['docs'], k) for i in range(len(query_ids)))
        meanPrecision = total_precision / len(query_ids) if len(query_ids) > 0 else 0
        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        relevant_docs = set(true_doc_IDs)
        retrieved_docs = query_doc_IDs_ordered[:k]
        num_relevant_retrieved = len(relevant_docs.intersection(retrieved_docs))
        recall = num_relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        total_recall = sum(self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrels[i]['docs'], k) for i in range(len(query_ids)))
        meanRecall = total_recall / len(query_ids) if len(query_ids) > 0 else 0
        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        total_fscore = sum(self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrels[i]['docs'], k) for i in range(len(query_ids)))
        meanFscore = total_fscore / len(query_ids) if len(query_ids) > 0 else 0
        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        DCG = 0
        relevant_docs = set(true_doc_IDs)
        for i in range(k):
            doc_id = query_doc_IDs_ordered[i]
            if doc_id in relevant_docs:
                DCG += 1 / math.log2(i + 2)  # i+2 because index starts from 0
        IDCG = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_docs))))
        nDCG = DCG / IDCG if IDCG > 0 else 0
        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        total_ndcg = sum(self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrels[i]['docs'], k) for i in range(len(query_ids)))
        meanNDCG = total_ndcg / len(query_ids) if len(query_ids) > 0 else 0
        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        relevant_docs = set(true_doc_IDs)
        num_relevant_docs = len(relevant_docs)
        cumulative_precision = 0
        num_relevant_retrieved = 0
        for i in range(k):
            doc_id = query_doc_IDs_ordered[i]
            if doc_id in relevant_docs:
                num_relevant_retrieved += 1
                cumulative_precision += num_relevant_retrieved / (i + 1)
        avgPrecision = cumulative_precision / num_relevant_docs if num_relevant_docs > 0 else 0
        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        total_avg_precision = sum(self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], qrels[i]['docs'], k) for i in range(len(query_ids)))
        meanAveragePrecision = total_avg_precision / len(query_ids) if len(query_ids) > 0 else 0
        return meanAveragePrecision
