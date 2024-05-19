#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <mutex>

std::mutex mtx;

int numOfThreads = -1;

struct GuessEvaluator {
    std::string target;

    float Evaluate(const std::string &guess) const {
        float sum = 0;
        for (int c = 0; c < std::min(target.size(), guess.size()); c++) {
            const float diff = std::fabs(target[c] - guess[c]);
            sum += diff * 256;
        }
        const float diffInLen = std::abs(int(guess.size()) - int(target.size()));
        const float totalDiff = sum + diffInLen * 256 * 256;
        assert(totalDiff >= 0.f);
        return totalDiff;
    }
};

struct GAParams {
    int generationSize = 500;
    int eliteCount = 10;
    int crossOverCount = 200;
    int mutatedCount = 200;
    float mutationRate = 0.05f;
    int individualSize = 300;
};

struct GA {
    struct Individual {
        std::string data;
        float diff = -1.f;
    };
    std::vector<Individual> generation;
    std::mt19937 rng;
    GuessEvaluator &eval;
    GAParams params;
    std::string allowedSymbols;
    std::mutex mtx;

    GA(GuessEvaluator &eval, GAParams params) : rng(42), eval(eval), params(params) {
        InitSymbols();
        for (int c = 0; c < params.generationSize; c++) {
            generation.push_back(RandomIndividual());
        }
    }

    void InitSymbols() {
        const char symbols[] = "=_!@#$%^&*()<>[];:'\" \n";
        allowedSymbols.reserve(256);
        for (int c = 'a'; c <= 'z'; c++) {
            allowedSymbols.push_back(char(c));
        }
        for (int c = 'A'; c <= 'Z'; c++) {
            allowedSymbols.push_back(char(c));
        }
        for (int c = '0'; c <= '9'; c++) {
            allowedSymbols.push_back(char(c));
        }

        for (int c = 0; c < std::size(symbols); c++) {
            allowedSymbols.push_back(symbols[c]);
        }
    }

    void Run(int maxGenerations) {
        std::vector<Individual> nextGeneration;
        for (int c = 0; c < maxGenerations; c++) {
            RankIndividuals();

            if (c % 1000 == 0)
                std::cout << generation[0].diff << ": " << generation[0].data << std::endl;
            nextGeneration.reserve(generation.size());

            for (int index = 0; index < params.eliteCount; index++) {
                nextGeneration.push_back(generation[index]);
            }

            for (int index = 0; index < params.crossOverCount; index++) {
                std::uniform_int_distribution<int> individualPicker(0, generation.size() - 1);
                const Individual &a = generation[individualPicker(rng)];
                const Individual &b = generation[individualPicker(rng)];
                nextGeneration.push_back(CrossOver(a, b));
            }

            std::uniform_int_distribution<int> individualPicker(0, nextGeneration.size() - 1);
            for (int index = 0; index < params.mutatedCount; index++) {
                const Individual &source = nextGeneration[individualPicker(rng)];
                nextGeneration.push_back(Mutate(source));
            }

            while (nextGeneration.size() < params.generationSize) {
                nextGeneration.push_back(RandomIndividual());
            }

            generation.swap(nextGeneration);
            nextGeneration.clear();
        }
    }

    void RankIndividuals() {
        for (int c = 0; c < generation.size(); c++) {
            generation[c].diff = eval.Evaluate(generation[c].data);
        }
        std::sort(generation.begin(), generation.end(), [this](const Individual &a, const Individual &b) {
            return a.diff < b.diff;
        });
    }

    Individual CrossOver(const Individual &a, const Individual &b) {
        const int newLen = (a.data.size() + b.data.size()) / 2;
        Individual result = a.data.size() > b.data.size() ? a : b;
        result.data.resize(newLen);
        std::discrete_distribution<> parentChooser({ double(2 + b.diff), double(2 + a.diff) });
        const Individual *parentChoose[2] = { &a, &b };

        for (int c = 0; c < std::min(a.data.size(), b.data.size()); c++) {
            result.data[c] = parentChoose[parentChooser(rng)]->data[c];
        }
        return result;
    }

    Individual Mutate(const Individual &source) {
        Individual mutated = source;

        std::uniform_real_distribution<float> mutateCheck(0, 1);
        std::uniform_int_distribution<int> letterDist(0, int(allowedSymbols.size() - 1));

        std::uniform_int_distribution<int> lengthChange(1 - source.data.size(), params.individualSize - source.data.size());
        const int newLength = lengthChange(rng) + source.data.size();
        mutated.data.resize(newLength, 'a');

        for (int c = source.data.size() - 1; c < mutated.data.size(); c++) {
            mutated.data[c] = allowedSymbols[letterDist(rng)];
        }

        for (int c = 0; c < mutated.data.size(); c++) {
            if (mutateCheck(rng) < params.mutationRate) {
                mutated.data[c] = allowedSymbols[letterDist(rng)];
            }
        }
        return mutated;
    }

    Individual RandomIndividual() {
        std::uniform_int_distribution<int> lenDist(1, 30);
        std::uniform_int_distribution<int> letterDist(0, int(allowedSymbols.size() - 1));

        Individual i;
        const int length = lenDist(rng);
        for (int c = 0; c < length; c++) {
            i.data.push_back(allowedSymbols[letterDist(rng)]);
        }
        return i;
    }
};


int main() {
    GuessEvaluator eval{ R"(struct GAParams {
    int generationSize = 500;
    int eliteCount = 10;
    int crossOverCount = 200;
    int mutatedCount = 200;
    float mutationRate = 0.05f;
};
)"};
    GAParams params{.individualSize=int(eval.target.size() * 2)};
    GA ga(eval, params);
    ga.Run(100'000'000);
    return 0;
}
