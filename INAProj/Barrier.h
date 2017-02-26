#pragma once
#include <mutex>
#include <condition_variable>
using namespace std;
class Barrier {
public:
	explicit Barrier(int iCount) :
		mThreshold(iCount),
		mCount(iCount),
		mGeneration(0) {
	}

	void Wait() {
		auto lGen = mGeneration;
		std::unique_lock<std::mutex> lLock{ mMutex };
		if (!--mCount) {
			mGeneration++;
			mCount = mThreshold;
			mCond.notify_all();
		}
		else {
			mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
		}
	}

private:
	std::mutex mMutex;
	std::condition_variable mCond;
	std::size_t mThreshold;
	std::size_t mCount;
	std::size_t mGeneration;
};