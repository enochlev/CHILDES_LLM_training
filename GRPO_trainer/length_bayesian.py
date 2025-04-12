class BayesianSentenceLengthSkewModel:
    def __init__(self):
        self.word_counts = []
        self.median_length = None
        self.left_variance = 1.0  # For values below median
        self.right_variance = 1.0  # For values above median
    
    def _count_words_till_punctuation(self, text):
        """Count words in each sentence and return their average."""
        punctuations = ['.', '!', '?']
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in punctuations:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence.strip())
        
        word_counts = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                word_counts.append(len(words))
        
        if word_counts:
            return sum(word_counts) / len(word_counts)
        return 0
    
    def fit(self, texts):
        """Fit the model to the dataset."""
        # Process each text
        for text in texts:
            count = self._count_words_till_punctuation(text)
            if count > 0:
                self.word_counts.append(count)
        
        # Calculate median
        if self.word_counts:
            self.word_counts.sort()
            if len(self.word_counts) % 2 == 0:
                self.median_length = (self.word_counts[len(self.word_counts)//2-1] + 
                                    self.word_counts[len(self.word_counts)//2]) / 2
            else:
                self.median_length = self.word_counts[len(self.word_counts)//2]
            
            # Calculate separate variances for left and right sides
            left_counts = [x for x in self.word_counts if x <= self.median_length]
            right_counts = [x for x in self.word_counts if x >= self.median_length]
            
            # Calculate left variance
            if left_counts:
                left_squared_diff = sum((x - self.median_length)**2 for x in left_counts)
                self.left_variance = left_squared_diff / len(left_counts)
                self.left_variance = max(self.left_variance, 0.5)  # Minimum variance
            
            # Calculate right variance
            if right_counts:
                right_squared_diff = sum((x - self.median_length)**2 for x in right_counts)
                self.right_variance = right_squared_diff / len(right_counts)
                self.right_variance = max(self.right_variance, 0.5)  # Minimum variance
        
        return self
    
    def predict(self, text, temperature=1.0):
        """Predict a score for the given text.
        
        Args:
            text: Input text to evaluate
            temperature: Controls the contrast of scores
                         temperature > 1.0: Amplifies high scores, reduces low scores
                         temperature = 1.0: No change (default behavior)
                         temperature < 1.0: Makes scores more uniform
        """
        if self.median_length is None:
            return 0.0
            
        avg_length = self._count_words_till_punctuation(text)
        
        # Choose variance based on which side of the median we're on
        if avg_length <= self.median_length:
            variance = self.left_variance
        else:
            variance = self.right_variance
        
        # Calculate base score using the appropriate variance
        distance = (avg_length - self.median_length)**2
        base_score = 1.0 * (2.71828 ** (-distance / (2 * variance)))
        
        # Apply temperature adjustment
        if temperature == 1.0:
            return base_score
        
        
        adjusted_score = base_score * (temperature ** (1 - base_score))
            
        return adjusted_score
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'w') as f:
            f.write(f"median_length={self.median_length}\n")
            f.write(f"left_variance={self.left_variance}\n")
            f.write(f"right_variance={self.right_variance}\n")
            f.write("word_counts=" + ",".join(map(str, self.word_counts)))
        
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        model = cls()
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("median_length="):
                    model.median_length = float(line.split("=")[1])
                elif line.startswith("left_variance="):
                    model.left_variance = float(line.split("=")[1])
                elif line.startswith("right_variance="):
                    model.right_variance = float(line.split("=")[1])
                elif line.startswith("word_counts="):
                    counts_str = line.split("=")[1]
                    model.word_counts = [float(x) for x in counts_str.split(",") if x]

        return model