# moodycamel::ConcurrentQueue<T>

An industrial-strength lock-free queue for C++.

Note: If all you need is a single-producer, single-consumer queue, I have [one of those too][spsc].

## Features

- Knock-your-socks-off [blazing fast performance][benchmarks].
- Single-header implementation. Just drop it in your project.
- Fully thread-safe lock-free queue. Use concurrently from any number of threads.
- C++11 implementation -- elements are moved (instead of copied) where possible.
- Templated, obviating the need to deal exclusively with pointers -- memory is managed for you.
- No artificial limitations on element types or maximum count.
- Memory can be allocated once up-front, or dynamically as needed.
- Fully portable (no assembly; all is done through standard C++11 primitives).
- Supports super-fast bulk operations.
- Includes a low-overhead blocking version (BlockingConcurrentQueue).
- Exception safe.

## Reasons to use

There are not that many full-fledged lock-free queues for C++. Boost has one, but it's limited to objects with trivial
assignment operators and trivial destructors, for example. Intel's TBB queue isn't lock-free, and requires trivial constructors too.
There's many academic papers that implement lock-free queues in C++, but usable source code is
hard to find, and tests even more so.

This queue not only has less limitations than others (for the most part), but [it's also faster][benchmarks].
It's been fairly well-tested, and offers advanced features like **bulk enqueueing/dequeueing**
(which, with my new design, is much faster than one element at a time, approaching and even surpassing
the speed of a non-concurrent queue even under heavy contention).

In short, there was a lock-free queue shaped hole in the C++ open-source universe, and I set out
to fill it with the fastest, most complete, and well-tested design and implementation I could.
The result is `moodycamel::ConcurrentQueue` :-)

## Reasons *not* to use

The fastest synchronization of all is the kind that never takes place. Fundamentally,
concurrent data structures require some synchronization, and that takes time. Every effort
was made, of course, to minimize the overhead, but if you can avoid sharing data between
threads, do so!

Why use concurrent data structures at all, then? Because they're gosh darn convenient! (And, indeed,
sometimes sharing data concurrently is unavoidable.)

My queue is **not linearizable** (see the next section on high-level design). The foundations of
its design assume that producers are independent; if this is not the case, and your producers
co-ordinate amongst themselves in some fashion, be aware that the elements won't necessarily
come out of the queue in the same order they were put in *relative to the ordering formed by that co-ordination*
(but they will still come out in the order they were put in by any *individual* producer). If this affects
your use case, you may be better off with another implementation; either way, it's an important limitation
to be aware of.

My queue is also **not NUMA aware**, and does a lot of memory re-use internally, meaning it probably doesn't
scale particularly well on NUMA architectures; however, I don't know of any other lock-free queue that *is*
NUMA aware (except for [SALSA][salsa], which is very cool, but has no publicly available implementation that I know of).

Finally, the queue is **not sequentially consistent**; there *is* a happens-before relationship between when an element is put
in the queue and when it comes out, but other things (such as pumping the queue until it's empty) require more thought
to get right in all eventualities, because explicit memory ordering may have to be done to get the desired effect. In other words,
it can sometimes be difficult to use the queue correctly. This is why it's a good idea to follow the [samples][samples.md] where possible.
On the other hand, the upside of this lack of sequential consistency is better performance.

## High-level design

Elements are stored internally using contiguous blocks instead of linked lists for better performance.
The queue is made up of a collection of sub-queues, one for each producer. When a consumer
wants to dequeue an element, it checks all the sub-queues until it finds one that's not empty.
All of this is largely transparent to the user of the queue, however -- it mostly just works<sup>TM</sup>.

One particular consequence of this design, however, (which seems to be non-intuitive) is that if two producers
enqueue at the same time, there is no defined ordering between the elements when they're later dequeued.
Normally this is fine, because even with a fully linearizable queue there'd be a race between the producer
threads and so you couldn't rely on the ordering anyway. However, if for some reason you do extra explicit synchronization
between the two producer threads yourself, thus defining a total order between enqueue operations, you might expect
that the elements would come out in the same total order, which is a guarantee my queue does not offer. At that
point, though, there semantically aren't really two separate producers, but rather one that happens to be spread
across multiple threads. In this case, you can still establish a total ordering with my queue by creating
a single producer token, and using that from both threads to enqueue (taking care to synchronize access to the token,
of course, but there was already extra synchronization involved anyway).

I've written a more detailed [overview of the internal design][blog], as well as [the full
nitty-gritty details of the design][design], on my blog. Finally, the
[source][source] itself is available for perusal for those interested in its implementation.

## Basic use

The entire queue's implementation is contained in **one header**, [`concurrentqueue.h`][concurrentqueue.h].
Simply download and include that to use the queue. The blocking version is in a separate header,
[`blockingconcurrentqueue.h`][blockingconcurrentqueue.h], that depends on the first.
The implementation makes use of certain key C++11 features, so it requires a fairly recent compiler
(e.g. VS2012+ or g++ 4.8; note that g++ 4.6 has a known bug with `std::atomic` and is thus not supported).
The algorithm implementations themselves are platform independent.

Use it like you would any other templated queue, with the exception that you can use
it from many threads at once :-)

Simple example:

    #include "concurrentqueue.h"

    moodycamel::ConcurrentQueue<int> q;
    q.enqueue(25);

    int item;
    bool found = q.try_dequeue(item);
    assert(found && item == 25);

Description of basic methods:
- `ConcurrentQueue(size_t initialSizeEstimate)`
      Constructor which optionally accepts an estimate of the number of elements the queue will hold
- `enqueue(T&& item)`
      Enqueues one item, allocating extra space if necessary
- `try_enqueue(T&& item)`
      Enqueues one item, but only if enough memory is already allocated
- `try_dequeue(T& item)`
      Dequeues one item, returning true if an item was found or false if the queue appeared empty

Note that it is up to the user to ensure that the queue object is completely constructed before
being used by any other threads (this includes making the memory effects of construction
visible, possibly via a memory barrier). Similarly, it's important that all threads have
finished using the queue (and the memory effects have fully propagated) before it is
destructed.

There's usually two versions of each method, one "explicit" version that takes a user-allocated per-producer or
per-consumer token, and one "implicit" version that works without tokens. Using the explicit methods is almost
always faster (though not necessarily by a huge factor). Apart from performance, the primary distinction between them
is their sub-queue allocation behaviour for enqueue operations: Using the implicit enqueue methods causes an
automatically-allocated thread-local producer sub-queue to be allocated (it is marked for reuse once the thread exits).
Explicit producers, on the other hand, are tied directly to their tokens' lifetimes (and are also recycled as needed).

Full API (pseudocode):

	# Allocates more memory if necessary
	enqueue(item) : bool
	enqueue(prod_token, item) : bool
	enqueue_bulk(item_first, count) : bool
	enqueue_bulk(prod_token, item_first, count) : bool

	# Fails if not enough memory to enqueue
	try_enqueue(item) : bool
	try_enqueue(prod_token, item) : bool
	try_enqueue_bulk(item_first, count) : bool
	try_enqueue_bulk(prod_token, item_first, count) : bool

	# Attempts to dequeue from the queue (never allocates)
	try_dequeue(item&) : bool
	try_dequeue(cons_token, item&) : bool
	try_dequeue_bulk(item_first, max) : size_t
	try_dequeue_bulk(cons_token, item_first, max) : size_t

	# If you happen to know which producer you want to dequeue from
	try_dequeue_from_producer(prod_token, item&) : bool
	try_dequeue_bulk_from_producer(prod_token, item_first, max) : size_t

	# A not-necessarily-accurate count of the total number of elements
	size_approx() : size_t

## Blocking version

As mentioned above, a full blocking wrapper of the queue is provided that adds
`wait_dequeue` and `wait_dequeue_bulk` methods in addition to the regular interface.
This wrapper is extremely low-overhead, but slightly less fast than the non-blocking
queue (due to the necessary bookkeeping involving a lightweight semaphore).

There are also timed versions that allow a timeout to be specified (either in microseconds
or with a `std::chrono` object).

The only major caveat with the blocking version is that you must be careful not to
destroy the queue while somebody is waiting on it. This generally means you need to
know for certain that another element is going to come along before you call one of
the blocking methods. (To be fair, the non-blocking version cannot be destroyed while
in use either, but it can be easier to coordinate the cleanup.)

Blocking example:

    #include "blockingconcurrentqueue.h"

    moodycamel::BlockingConcurrentQueue<int> q;
    std::thread producer([&]() {
        for (int i = 0; i != 100; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(i % 10));
            q.enqueue(i);
        }
    });
    std::thread consumer([&]() {
        for (int i = 0; i != 100; ++i) {
            int item;
            q.wait_dequeue(item);
            assert(item == i);

            if (q.wait_dequeue_timed(item, std::chrono::milliseconds(5))) {
                ++i;
                assert(item == i);
            }
        }
    });
    producer.join();
    consumer.join();

    assert(q.size_approx() == 0);

## Advanced features

#### Tokens

The queue can take advantage of extra per-producer and per-consumer storage if
it's available to speed up its operations. This takes the form of "tokens":
You can create a consumer token and/or a producer token for each thread or task
(tokens themselves are not thread-safe), and use the methods that accept a token
as their first parameter:

    moodycamel::ConcurrentQueue<int> q;

    moodycamel::ProducerToken ptok(q);
    q.enqueue(ptok, 17);

    moodycamel::ConsumerToken ctok(q);
    int item;
    q.try_dequeue(ctok, item);
    assert(item == 17);

If you happen to know which producer you want to consume from (e.g. in
a single-producer, multi-consumer scenario), you can use the `try_dequeue_from_producer`
methods, which accept a producer token instead of a consumer token, and cut some overhead.

Note that tokens work with the blocking version of the queue too.

When producing or consuming many elements, the most efficient way is to:

1. Use the bulk methods of the queue with tokens
2. Failing that, use the bulk methods without tokens
3. Failing that, use the single-item methods with tokens
4. Failing that, use the single-item methods without tokens

Having said that, don't create tokens willy-nilly -- ideally there would be
one token (of each kind) per thread. The queue will work with what it is
given, but it performs best when used with tokens.

Note that tokens aren't actually tied to any given thread; it's not technically
required that they be local to the thread, only that they be used by a single
producer/consumer at a time.

#### Bulk operations

Thanks to the [novel design][blog] of the queue, it's just as easy to enqueue/dequeue multiple
items as it is to do one at a time. This means that overhead can be cut drastically for
bulk operations. Example syntax:

    moodycamel::ConcurrentQueue<int> q;

    int items[] = { 1, 2, 3, 4, 5 };
    q.enqueue_bulk(items, 5);

    int results[5];     // Could also be any iterator
    size_t count = q.try_dequeue_bulk(results, 5);
    for (size_t i = 0; i != count; ++i) {
        assert(results[i] == items[i]);
    }

#### Preallocation (correctly using `try_enqueue`)

`try_enqueue`, unlike just plain `enqueue`, will never allocate memory. If there's not enough room in the
queue, it simply returns false. The key to using this method properly, then, is to ensure enough space is
pre-allocated for your desired maximum element count.

The constructor accepts a count of the number of elements that it should reserve space for. Because the
queue works with blocks of elements, however, and not individual elements themselves, the value to pass
in order to obtain an effective number of pre-allocated element slots is non-obvious.

First, be aware that the count passed is rounded up to the next multiple of the block size. Note that the
default block size is 32 (this can be changed via the traits). Second, once a slot in a block has been
enqueued to, that slot cannot be re-used until the rest of the block has completely been completely filled
up and then completely emptied. This affects the number of blocks you need in order to account for the
overhead of partially-filled blocks. Third, each producer (whether implicit or explicit) claims and recycles
blocks in a different manner, which again affects the number of blocks you need to account for a desired number of
usable slots.

Suppose you want the queue to be able to hold at least `N` elements at any given time. Without delving too
deep into the rather arcane implementation details, here are some simple formulas for the number of elements
to request for pre-allocation in such a case. Note the division is intended to be arithmetic division and not
integer division (in order for `ceil()` to work).

For explicit producers (using tokens to enqueue):

    (ceil(N / BLOCK_SIZE) + 1) * MAX_NUM_PRODUCERS * BLOCK_SIZE

For implicit producers (no tokens):

    (ceil(N / BLOCK_SIZE) - 1 + 2 * MAX_NUM_PRODUCERS) * BLOCK_SIZE

When using mixed producer types:

    ((ceil(N / BLOCK_SIZE) - 1) * (MAX_EXPLICIT_PRODUCERS + 1) + 2 * (MAX_IMPLICIT_PRODUCERS + MAX_EXPLICIT_PRODUCERS)) * BLOCK_SIZE

If these formulas seem rather inconvenient, you can use the constructor overload that accepts the minimum
number of elements (`N`) and the maximum number of explicit and implicit producers directly, and let it do the
computation for you.

Finally, it's important to note that because the queue is only eventually consistent and takes advantage of
weak memory ordering for speed, there's always a possibility that under contention `try_enqueue` will fail
even if the queue is correctly pre-sized for the desired number of elements. (e.g. A given thread may think that
the queue's full even when that's no longer the case.) So no matter what, you still need to handle the failure
case (perhaps looping until it succeeds), unless you don't mind dropping elements.

#### Exception safety

The queue is exception safe, and will never become corrupted if used with a type that may throw exceptions.
The queue itself never throws any exceptions (operations fail gracefully (return false) if memory allocation
fails instead of throwing `std::bad_alloc`).

It is important to note that the guarantees of exception safety only hold if the element type never throws
from its destructor, and that any iterators passed into the queue (for bulk operations) never throw either.
Note that in particular this means `std::back_inserter` iterators must be used with care, since the vector
being inserted into may need to allocate and throw a `std::bad_alloc` exception from inside the iterator;
so be sure to reserve enough capacity in the target container first if you do this.

The guarantees are presently as follows:
- Enqueue operations are rolled back completely if an exception is thrown from an element's constructor.
  For bulk enqueue operations, this means that elements are copied instead of moved (in order to avoid
  having only some of the objects be moved in the event of an exception). Non-bulk enqueues always use
  the move constructor if one is available.
- If the assignment operator throws during a dequeue operation (both single and bulk), the element(s) are
  considered dequeued regardless. In such a case, the dequeued elements are all properly destructed before
  the exception is propagated, but there's no way to get the elements themselves back.
- Any exception that is thrown is propagated up the call stack, at which point the queue is in a consistent
  state.

Note: If any of your type's copy constructors/move constructors/assignment operators don't throw, be sure
to annotate them with `noexcept`; this will avoid the exception-checking overhead in the queue where possible
(even with zero-cost exceptions, there's still a code size impact that has to be taken into account).

#### Traits

The queue also supports a traits template argument which defines various types, constants,
and the memory allocation and deallocation functions that are to be used by the queue. The typical pattern
to providing your own traits is to create a class that inherits from the default traits
and override only the values you wish to change. Example:

    struct MyTraits : public moodycamel::ConcurrentQueueDefaultTraits
    {
	static const size_t BLOCK_SIZE = 256;		// Use bigger blocks
    };

    moodycamel::ConcurrentQueue<int, MyTraits> q;

#### How to dequeue types without calling the constructor

The normal way to dequeue an item is to pass in an existing object by reference, which
is then assigned to internally by the queue (using the move-assignment operator if possible).
This can pose a problem for types that are
expensive to construct or don't have a default constructor; fortunately, there is a simple
workaround: Create a wrapper class that copies the memory contents of the object when it
is assigned by the queue (a poor man's move, essentially). Note that this only works if
the object contains no internal pointers. Example:

    struct MyObjectMover
    {
        inline void operator=(MyObject&& obj)
        {
            std::memcpy(data, &obj, sizeof(MyObject));

            // TODO: Cleanup obj so that when it's destructed by the queue
            // it doesn't corrupt the data of the object we just moved it into
        }

        inline MyObject& obj() { return *reinterpret_cast<MyObject*>(data); }

    private:
	align(alignof(MyObject)) char data[sizeof(MyObject)];
    };

## Samples

There are some more detailed samples [here][samples.md]. The source of
the [unit tests][unittest-src] and [benchmarks][benchmark-src] are available for reference as well.

## Benchmarks

See my blog post for some [benchmark results][benchmarks] (including versus `boost::lockfree::queue` and `tbb::concurrent_queue`),
or run the benchmarks yourself (requires MinGW and certain GnuWin32 utilities to build on Windows, or a recent
g++ on Linux):

    cd build
    make benchmarks
    bin/benchmarks

The short version of the benchmarks is that it's so fast (especially the bulk methods), that if you're actually
using the queue to *do* anything, the queue won't be your bottleneck.

## Tests (and bugs)

I've written quite a few unit tests as well as a randomized long-running fuzz tester. I also ran the
core queue algorithm through the [CDSChecker][cdschecker] C++11 memory model model checker. Some of the
inner algorithms were tested separately using the [Relacy][relacy] model checker, and full integration
tests were also performed with Relacy.
I've tested
on Linux (Fedora 19) and Windows (7), but only on x86 processors so far (Intel and AMD). The code was
written to be platform-independent, however, and should work across all processors and OSes.

Due to the complexity of the implementation and the difficult-to-test nature of lock-free code in general,
there may still be bugs. If anyone is seeing buggy behaviour, I'd like to hear about it! (Especially if
a unit test for it can be cooked up.) Just open an issue on GitHub.

## License

I'm releasing the source of this repository (with the exception of third-party code, i.e. the Boost queue
(used in the benchmarks for comparison), Intel's TBB library (ditto), CDSChecker, Relacy, and Jeff Preshing's
cross-platform semaphore, which all have their own licenses)
under a simplified BSD license. I'm also dual-licensing under the Boost Software License.
See the [LICENSE.md][license] file for more details.

Note that lock-free programming is a patent minefield, and this code may very
well violate a pending patent (I haven't looked), though it does not to my present knowledge.
I did design and implement this queue from scratch.

## Diving into the code

If you're interested in the source code itself, it helps to have a rough idea of how it's laid out. This
section attempts to describe that.

The queue is formed of several basic parts (listed here in roughly the order they appear in the source). There's the
helper functions (e.g. for rounding to a power of 2). There's the default traits of the queue, which contain the
constants and malloc/free functions used by the queue. There's the producer and consumer tokens. Then there's the queue's
public API itself, starting with the constructor, destructor, and swap/assignment methods. There's the public enqueue methods,
which are all wrappers around a small set of private enqueue methods found later on. There's the dequeue methods, which are
defined inline and are relatively straightforward.

Then there's all the main internal data structures. First, there's a lock-free free list, used for recycling spent blocks (elements
are enqueued to blocks internally). Then there's the block structure itself, which has two different ways of tracking whether
it's fully emptied or not (remember, given two parallel consumers, there's no way to know which one will finish first) depending on where it's used.
Then there's a small base class for the two types of internal SPMC producer queues (one for explicit producers that holds onto memory
but attempts to be faster, and one for implicit ones which attempt to recycle more memory back into the parent but is a little slower).
The explicit producer is defined first, then the implicit one. They both contain the same general four methods: One to enqueue, one to
dequeue, one to enqueue in bulk, and one to dequeue in bulk. (Obviously they have constructors and destructors too, and helper methods.)
The main difference between them is how the block handling is done (they both use the same blocks, but in different ways, and map indices
to them in different ways).

Finally, there's the miscellaneous internal methods: There's the ones that handle the initial block pool (populated when the queue is constructed),
and an abstract block pool that comprises the initial pool and any blocks on the free list. There's ones that handle the producer list
(a lock-free add-only linked list of all the producers in the system). There's ones that handle the implicit producer lookup table (which
is really a sort of specialized TLS lookup). And then there's some helper methods for allocating and freeing objects, and the data members
of the queue itself, followed lastly by the free-standing swap functions.


[blog]: http://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++
[design]: http://moodycamel.com/blog/2014/detailed-design-of-a-lock-free-queue
[samples.md]: https://github.com/cameron314/concurrentqueue/blob/master/samples.md
[source]: https://github.com/cameron314/concurrentqueue
[concurrentqueue.h]: https://github.com/cameron314/concurrentqueue/blob/master/concurrentqueue.h
[blockingconcurrentqueue.h]: https://github.com/cameron314/concurrentqueue/blob/master/blockingconcurrentqueue.h
[unittest-src]: https://github.com/cameron314/concurrentqueue/tree/master/tests/unittests
[benchmarks]: http://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++#benchmarks
[benchmark-src]: https://github.com/cameron314/concurrentqueue/tree/master/benchmarks
[license]: https://github.com/cameron314/concurrentqueue/blob/master/LICENSE.md
[cdschecker]: http://demsky.eecs.uci.edu/c11modelchecker.html
[relacy]: http://www.1024cores.net/home/relacy-race-detector
[spsc]: https://github.com/cameron314/readerwriterqueue
[salsa]: http://webee.technion.ac.il/~idish/ftp/spaa049-gidron.pdf
