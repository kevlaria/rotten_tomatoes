/**
 * Created by kevinlee on 27/1/15.
 */

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;


public class Analyser {

        public int findSentiment(String line) {

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            int mainSentiment = 0;
            if (line != null && line.length() > 0) {
                int longest = 0;
                Annotation annotation = pipeline.process(line);
                System.out.println("Annotation: " + annotation);
                for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                    Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
//                    System.out.println(tree);
//                    System.out.println("Label: " + tree.children()[0].label());
                    int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
//                    System.out.println(RNNCoreAnnotations.getPredictions(tree));
//                    System.out.println("Total Sentiment: " + sentiment);
                    String partText = sentence.toString();
                    if (partText.length() > longest) {
                        mainSentiment = sentiment;
                        longest = partText.length();
                    }

                }
            }

            return mainSentiment;

        }

        private String toCss(int sentiment) {
            switch (sentiment) {
                case 0:
                    return "0";
                case 1:
                    return "1";
                case 2:
                    return "2";
                case 3:
                    return "3";
                case 4:
                    return "4";
                default:
                    return "";
            }
        }

        public static void main(String[] args) throws IOException {
            Analyser sentimentAnalyzer = new Analyser();

            List<String> lines = sentimentAnalyzer.openFile();
            lines.remove(0);

            List<String> ids = new ArrayList<String>();
            List<String> phrases = new ArrayList<String>();

            List<Integer> sentiment = new ArrayList<Integer>();


            for (String line : lines){
                String[] portions = line.split("\t");
                ids.add(portions[0]);
                phrases.add(portions[2]);
            }

            for (String phrase : phrases){
                int sentimentValue = sentimentAnalyzer.findSentiment(phrase);
                sentiment.add(sentimentValue);
            }

            sentimentAnalyzer.writeToFile(ids, sentiment);


        }


        private List<String> openFile() throws IOException {
            BufferedReader reader = new BufferedReader(new FileReader("data/test.tsv"));
            List<String> contents = new ArrayList<String>();
            String line;
            while ((line = reader.readLine()) != null){
                contents.add(line);
            }
            reader.close();
            return contents;
        }

        private void writeToFile(List<String> ids, List<Integer> sentiment) throws IOException {
            BufferedWriter writer = new BufferedWriter(new FileWriter("data/predictions.tsv"));
            writer.write("PhraseId,Sentiment\n");
            for (int i = 0; i < ids.size(); i++){
                writer.write(ids.get(i) + "," + sentiment.get(i) + "\n");
            }
            writer.close();
        }


}
