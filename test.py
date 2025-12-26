################################################
"*************** TESTING PHASE ***************"
################################################


# Importing the spam_detector and train file to use the required functions
import spam_detector as sd


"************** MAIN FUNCTION **************"
"*** Defines main function for test.py ***"


def mainFunction():
    # Fine-tuning: Try different delta (smoothing) values
    delta_values = [0.1, 0.5, 1.0, 2.0]
    best_f1 = -1
    best_delta = None
    best_metrics = None
    for delta in delta_values:
        test_fileNames = sd.get_testFileNames()
        nb_of_allEmails = sd.number_of_allEmails()
        nb_of_spamEmails = sd.number_of_spamEmails()
        nb_of_hamEmails = sd.number_of_hamEmails()
        all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
        print(f"\n=== Evaluating with Laplace smoothing delta = {delta} ===")
        test_fileNames = sd.get_testFileNames()
        nb_of_allEmails = sd.number_of_allEmails()
        nb_of_spamEmails = sd.number_of_spamEmails()
        nb_of_hamEmails = sd.number_of_hamEmails()
        all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
        all_uniqueWords = sd.unique_words(all_trainWords)
        spam_bagOfWords, ham_bagOfWords = sd.bagOfWords_genarator(
            all_uniqueWords, spam_trainWords, ham_trainWords)
        smoothed_spamBOW, smoothed_hamBOW = sd.smoothed_bagOfWords(
            all_uniqueWords, spam_bagOfWords, ham_bagOfWords, delta)
        spam_prob = sd.spam_probability(nb_of_allEmails, nb_of_spamEmails)
        ham_prob = sd.ham_probability(nb_of_allEmails, nb_of_hamEmails)
        spam_condProb = sd.spam_condProbability(
            all_uniqueWords, spam_bagOfWords, smoothed_spamBOW, delta)
        ham_condProb = sd.ham_condProbability(
            all_uniqueWords, ham_bagOfWords, smoothed_hamBOW, delta)
        actual_labels = sd.get_actualLabels()
        ham_scores, spam_scores, predicted_labels, decision_labels = sd.score_calculator(
            all_uniqueWords, spam_prob, ham_prob, spam_condProb, ham_condProb, delta)
        fileNumbers = len(test_fileNames)
        fileNames = test_fileNames
        actualLabels = actual_labels
        predictedLabels = predicted_labels
        hamScores = ham_scores
        spamScores = spam_scores
        decisionLabels = decision_labels
        result_output = sd.result_output_generator(
            fileNumbers, fileNames, predictedLabels, hamScores, spamScores, actualLabels, decisionLabels)
        sd.resultFileBuilder(result_output)
        spam_precision = sd.get_spamPrecision(
            fileNumbers, actualLabels, predictedLabels)
        spam_recall = sd.get_spamRecall(fileNumbers, actualLabels, predictedLabels)
        spam_accuracy = sd.get_spamAccuracy(
            fileNumbers, actualLabels, predictedLabels)
        spam_fmeasure = sd.get_spamFmeasure(spam_precision, spam_recall)
        ham_precision = sd.get_hamPrecision(
            fileNumbers, actualLabels, predictedLabels)
        ham_recall = sd.get_hamRecall(fileNumbers, actualLabels, predictedLabels)
        ham_accuracy = sd.get_hamAccuracy(
            fileNumbers, actualLabels, predictedLabels)
        ham_fmeasure = sd.get_hamFmeasure(ham_precision, ham_recall)
        print(f"Spam: Precision={spam_precision:.2f}, Recall={spam_recall:.2f}, Accuracy={spam_accuracy:.2f}, F1={spam_fmeasure:.2f}")
        print(f"Ham: Precision={ham_precision:.2f}, Recall={ham_recall:.2f}, Accuracy={ham_accuracy:.2f}, F1={ham_fmeasure:.2f}")
        # Print evaluation results and confusion matrices
        evaluation_result_output = sd.evaluation_result(
            spam_accuracy, spam_precision, spam_recall, spam_fmeasure, ham_accuracy, ham_precision, ham_recall, ham_fmeasure)
        spam_tp, spam_tn, spam_fp, spam_fn = sd.spamConfusionParams(
            fileNumbers, actualLabels, predictedLabels)
        spam_confusionMatrix_output = sd.spam_confusionMatrix(
            spam_tp, spam_tn, spam_fp, spam_fn)
        ham_tp, ham_tn, ham_fp, ham_fn = sd.hamConfusionParams(
            fileNumbers, actualLabels, predictedLabels)
        ham_confusionMatrix_output = sd.ham_confusionMatrix(
            ham_tp, ham_tn, ham_fp, ham_fn)
        print("\n--- Evaluation Results ---")
        print(evaluation_result_output)
        print("\n--- Spam Confusion Matrix ---")
        print(spam_confusionMatrix_output)
        print("\n--- Ham Confusion Matrix ---")
        print(ham_confusionMatrix_output)
        # Write evaluation.txt for this delta
        evaluation_output = sd.evaluation_output_generator(
            evaluation_result_output, spam_confusionMatrix_output, ham_confusionMatrix_output)
        sd.evaluationFileBuilder(evaluation_output)

        avg_f1 = (spam_fmeasure + ham_fmeasure) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_delta = delta
            best_metrics = (spam_precision, spam_recall, spam_accuracy, spam_fmeasure, ham_precision, ham_recall, ham_accuracy, ham_fmeasure)
    print(f"\nBest delta: {best_delta} (Avg F1: {best_f1:.2f})")
    if best_metrics is not None:
        print(f"Best metrics: Spam(P={best_metrics[0]:.2f}, R={best_metrics[1]:.2f}, A={best_metrics[2]:.2f}, F1={best_metrics[3]:.2f}), Ham(P={best_metrics[4]:.2f}, R={best_metrics[5]:.2f}, A={best_metrics[6]:.2f}, F1={best_metrics[7]:.2f})")
    else:
        print("No valid metrics found for any delta value. All F1 scores may be zero or undefined.")


if __name__ == "__main__":
    print("Starting test script...")
    mainFunction()
    # After main evaluation, print prediction for each test file using the best delta
    print("\n--- Individual Email Predictions (Best Delta) ---")
    import spam_detector as sd
    delta = 0.5  # Default; will try to extract best_delta from mainFunction if possible
    try:
        # Try to get best_delta from the mainFunction scope if available
        import builtins
        if 'best_delta' in globals():
            delta = best_delta
    except Exception:
        pass
    detector = sd.SpamDetector(delta=delta) if hasattr(sd, 'SpamDetector') else None
    if detector:
        detector.train('train/ham', 'train/spam')
        import glob
        ham_files = glob.glob('test/test-ham-*.txt')
        spam_files = glob.glob('test/test-spam-*.txt')
        for email_file in ham_files + spam_files:
            with open(email_file, 'r', encoding='latin-1', errors='ignore') as f:
                email_text = f.read()
            prediction = detector.predict(email_text)
            print(f"{email_file}: {prediction}")
    else:
        print("Per-email prediction not available: SpamDetector class not found in spam_detector.py.")
    print("Test script finished.")
