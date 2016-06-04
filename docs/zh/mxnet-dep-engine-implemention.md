<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#orgheadline1">1. 基本背景知识</a></li>
<li><a href="#orgheadline2">2. Op</a></li>
<li><a href="#orgheadline9">3. Var</a>
<ul>
<li><a href="#orgheadline3">3.1. 类图</a></li>
<li><a href="#orgheadline4">3.2. 理解Var的队列</a></li>
<li><a href="#orgheadline5">3.3. 添加读依赖</a></li>
<li><a href="#orgheadline6">3.4. 添加写依赖</a></li>
<li><a href="#orgheadline7">3.5. 读依赖完成</a></li>
<li><a href="#orgheadline8">3.6. 写依赖完成</a></li>
</ul>
</li>
<li><a href="#orgheadline11">4. Engine</a>
<ul>
<li><a href="#orgheadline10">4.1. 总结</a></li>
</ul>
</li>
</ul>
</div>
</div>


# 基本背景知识<a id="orgheadline1"></a>

MXNET中有一个依赖引擎，这个引擎是用来分析计算过程的依赖关系，把不依赖的计算并行
化，以达到提高性能的目的。它的基本原理可以看官方的[文档](https://mxnet-bing.readthedocs.io/en/latest/system/note_engine.html)。 简单的说就是给每一个对
象打上一个tag，这个tag叫做Var，每一个计算(op)都会依赖一个或者多个Var，依赖有两种
类型：写依赖和读依赖。依赖引擎为每一个Var都维护一个队列，然后根据op的依赖关系向
队列中添加ReadDependency和WriteDependency，当各个依赖完成后要更新队列的状态。

# Op<a id="orgheadline2"></a>

Op实际上是用来代表计算过程以及它依赖的var，先来看看它的uml类图。

![img](http://yuyang0.github.io/articles/static/img/opr-class-uml.png)

上面一些比较重要的属性如下：

1.  fn： op实际要执行的函数
2.  const\_vars, mutable\_vars: 依赖的var列表（读和写）。
3.  wait: 当前还没有就绪的var的个数，它的初始值是
    `len(const_vars)+len(mutable_vars)`, 每一个依赖就绪那么就会调用 `dec_wait` 将
    该值减一，如果该值为0，那么所有的依赖都已就绪，那么可以丢到执行引擎执行了。

# Var<a id="orgheadline9"></a>

var可以看做是一个tag，用来标示每一个对象的，这样Op对对象的依赖可以简化成对var的
依赖，这样就可以构建出一个不依赖于具体的对象的通用的依赖引擎。Var是依赖引擎的关键。

## 类图<a id="orgheadline3"></a>

![img](http://yuyang0.github.io/articles/static/img/threaded-var-class-uml.png)

**声明：下文说到执行时，意思是Op的当前var的依赖已经就绪，因为一个op可以依赖多个
var，如果其他的Var没有就绪，那么这时op可能并没有实际运行**

Var只是一个基类，用来统一类型系统的，主要的工作在 `ThreadedVar` 中，每一个对象都
会有一个由 `VersionedVarBlock` 所组成的链表，这个链表就是一个FIFO队列。 `head_`
指向的是队列的尾部, 实际是一个哨兵(空对象)， `head_` 这个命名有误导性,
`pending_write_` 指向的是最"老"的写依赖，如果没有写依赖，那么就指向 `nullptr`,
根据依赖引擎的特点，它实际上指向的是队列的头部， `ThreadedVar` 的那四个方法就是
来操作这个队列的。

1.  num\_pending\_reads\_: 代表当前正在执行(还没有执行完)的读依赖的个数
2.  pending\_write\_: 代表队列中最“老”的写依赖, 它一直指向队列的头部。
3.  head\_: 队列的尾部。

需要注意的是，正在执行的读依赖是不在队列中的，但是正在执行的写依赖是在队列中的。

## 理解Var的队列<a id="orgheadline4"></a>

var的队列是依赖引擎的核心，下面我们来分析下各种情况下，如何修改队列的状态。

1.  添加读依赖: 如果前面没有写依赖，那么直接运行, 否则就插入队列的尾部(head\_那一端)
2.  添加写依赖： 直接将依赖插入队列的尾部，并检查是不是写就绪(既没有读依赖也没有
    写依赖在运行),如果是写就绪，那么就运行该依赖。
3.  读依赖完成
4.  写依赖完成

![img](http://yuyang0.github.io/articles/static/img/threaded-var-queue1.png)

上图中w1写依赖正在执行。

![img](http://yuyang0.github.io/articles/static/img/threaded-var-queue2.png)
写依赖w1完成将自己移出队列，并执行写依赖w2

![img](http://yuyang0.github.io/articles/static/img/threaded-var-queue3.png)

写依赖w2完成后将自己移出队列，接着并行的执行读依赖r1，r2，记住正在执行的读依赖是被移出队列的，
它们的数目使用 `num_pending_reads_` 跟踪的

![img](http://yuyang0.github.io/articles/static/img/threaded-var-queue4.png)

每一个读依赖完成都会将 `num_pending_reads_` 减一，如果减为了0，那么就意味着所有
的读依赖都完成了，当r1，r2都完成后，接着执行w3写依赖。

## 添加读依赖<a id="orgheadline5"></a>

代码主要在 `src/engine/Threaded_engine.cc` 的 `AppendReadDependency` 中。

```cpp
inline void ThreadedVar::AppendReadDependency(OprBlock* opr_block) {
    std::lock_guard<std::mutex> lock{m_};
    if (pending_write_ == nullptr) {
        // invariant: is_ready_to_read()
        CHECK_GE(num_pending_reads_, 0);
        // STATE CHANGE
        ++num_pending_reads_;
        // decrease wait counter
        opr_block->decr_wait();
    } else {
        auto&& new_var_block = VersionedVarBlock::New();
        assert(head_->next == nullptr);
        assert(head_->trigger == nullptr);
        assert(head_->write == false);
        // append things to next.
        head_->next = new_var_block;
        head_->trigger = opr_block;
        head_ = new_var_block;
    }
}
```

代码的基本思路是这样的：检查队列中有没有写依赖，这分两种情况：

1.  如果没有写依赖，那么意味着，目前该Var没有依赖在执行，或者说只有读依赖在执行，
    所以这个新的读依赖可以直接执行，那么它没有必要添加到队列中，只需要更新
    `num_pending_reads_` 就好，当然因为该op可能还依赖别的var，所以你只能调用
    `decr_wait` ，只有当wait减为0的时候，才能开始运行。这部分代码在engine的push中。
2.  如果有写依赖，那么读依赖必须在写依赖的后面执行，所以需要把读依赖添加到队列的
    尾部。记住 `head_` 永远指向一个空的哨兵对象。

## 添加写依赖<a id="orgheadline6"></a>

代码主要在 `src/engine/Threaded_engine.cc` 的 `AppendWriteDependency` 中。

```cpp
inline void ThreadedVar::AppendWriteDependency(OprBlock* opr_block) {
    auto&& new_var_block = VersionedVarBlock::New();
    std::lock_guard<std::mutex> lock{m_};
    // invariant.
    assert(head_->next == nullptr);
    assert(head_->trigger == nullptr);
    assert(head_->write == false);
    // attach to head.
    head_->next = new_var_block;
    head_->trigger = opr_block;
    head_->write = true;

    // check if it is ready to write
    if (pending_write_ == nullptr) {
        // invariant: is_ready_to_read()
        pending_write_ = head_;
        CHECK_GE(num_pending_reads_, 0);
        if (num_pending_reads_ == 0) {
            // STATE CHANGE
            opr_block->decr_wait();
            num_pending_reads_ = kWriteTriggered;
        }
    } else {
        CHECK_NE(num_pending_reads_, 0);
    }
    head_ = new_var_block;
}
```

代码的基本思路是这样的： 将该Op放入队列的尾部，接着检查该Op的依赖有没有就绪，这
要检查Var有没有写依赖(pending\_read\_==nullptr)和读依赖(num\_pending\_read\_==0)的Op
正在执行，只有二者都没有时，才能开始运行，当然你依然要检查该Op对其他的Var的依赖
有没有就绪。需要注意的一点是，即便Op的Var写依赖就绪，该Op也不会从队列中移除，只
有该Op执行完成后才会被移除，这在CompleteWriteDependency中实现。

## 读依赖完成<a id="orgheadline7"></a>

代码主要在 `src/engine/Threaded_engine.cc` 的 `CompleteReadDependency` 中。

```cpp
template <typename Dispatcher>
inline void ThreadedVar::CompleteReadDependency(Dispatcher dispatcher) {
    OprBlock *trigger = nullptr;
    {
        // this is lock scope
        std::lock_guard<std::mutex> lock{m_};
        CHECK_GT(num_pending_reads_, 0);

        if (--num_pending_reads_ == 0) {
            if (pending_write_ != nullptr) {
                // STATE CHANGE
                trigger = pending_write_->trigger;
                num_pending_reads_ = kWriteTriggered;
            }
        }
    }
    if (trigger != nullptr && trigger->decr_wait() == 0) {
        dispatcher(trigger);
    }
}
```

该部分代码会在一个op运算完成后调用，代码逻辑是比较简单的，先更新
`num_pending_read_`, 更新后如果该值为0，那么就意味着，所有的读依赖都已经执行完成,
这样就检查队列，若是存在写依赖，那么该写依赖就就绪了，那么Op就可以执行了(前提是
依赖的其他var也都就绪了, wait为0)。上面的dispatcher实际就是用来将Op丢入执行引擎
的，它一般是PushToExecute，这个后文会看到。

## 写依赖完成<a id="orgheadline8"></a>

代码主要在 `src/engine/Threaded_engine.cc` 的 `CompleteWriteDependency` 中。

```cpp
template <typename Dispatcher>
inline bool ThreadedVar::CompleteWriteDependency(Dispatcher dispatcher) {
  // this is lock scope
  VersionedVarBlock *old_pending_write, *end_of_read_chain;
  OprBlock* trigger_write = nullptr;
  {
    std::lock_guard<std::mutex> lock{m_};
    // invariants
    assert(head_->next == nullptr);
    assert(pending_write_ != nullptr);
    CHECK_EQ(num_pending_reads_, kWriteTriggered);

    // really delete
    if (to_delete_) {
      VersionedVarBlock *head = pending_write_->next;
      VersionedVarBlock::Delete(pending_write_);
      assert(head_ == head);
      VersionedVarBlock::Delete(head);
      return true;
    }
    // detach pending write
    old_pending_write = pending_write_;
    // search for chains to trigger
    end_of_read_chain = old_pending_write->next;
    // reset to 0 pending reads
    num_pending_reads_ = 0;
    while (end_of_read_chain != head_ &&
           end_of_read_chain->write == false) {
      ++num_pending_reads_;
      end_of_read_chain = end_of_read_chain->next;
    }
    if (end_of_read_chain == head_) {
      pending_write_ = nullptr;
    } else {
      // check if there is pending reads, if not trigger write
      assert(end_of_read_chain->write == true);
      pending_write_ = end_of_read_chain;
      if (num_pending_reads_ == 0) {
        // mark write as already actived in this var
        num_pending_reads_ = kWriteTriggered;
        trigger_write = end_of_read_chain->trigger;
      }
    }
  }
  // This is outside of lock scope
  // Be very carful, pending_write_ and num_pending_reads_
  // can change now, do not reply ont the two variables.
  // The linked list \in [old_pending_write, end_of_read_chain)
  // is already detached from this Var.
  // So it is safe to modify these
  VersionedVarBlock *cur_head = old_pending_write->next;
  VersionedVarBlock::Delete(old_pending_write);
  // dispatch all the events
  while (cur_head != end_of_read_chain) {
    if (cur_head->trigger->decr_wait() == 0) {
      dispatcher(cur_head->trigger);
    }
    auto prev = cur_head;
    cur_head = cur_head->next;
    assert(cur_head != nullptr);
    VersionedVarBlock::Delete(prev);
  }
  if (trigger_write != nullptr && trigger_write->decr_wait() == 0) {
    dispatcher(trigger_write);
  }
  return false;
}
```

和读依赖完成类似，只是写依赖的后面可能跟着多个读依赖，所以需要遍历链表直到发现下
一个写依赖, 这个写依赖由 `end_of_read_chain` 指针来表示，如果没发现写依赖，那么
该指针指向 `head_`,遍历的过程中每发现一个读依赖就将 `num_pending_reads_` 加一，
这样当遍历结束后， `old_pending_write` 指向已经完成的写依赖，而
`end_of_read_chain` 指向下一个写依赖或者 `head_`, 这时候有两种情况：

1.  这两个指针的中间有多个元素，很显然这是多个读依赖，第二个 `while` 循环就是用来
    并行的执行这两个指针中间的读依赖的。
2.  这两个指针之间没有元素，那么意味着没有读依赖，那么就直接执行
    `end_of_read_chian` 指向的写依赖，如果该指针指向 `head_` 那么意味着队列为空，
    什么也不用做。 最后 一部分的 `if` 就是用来处理这个情况的。

# Engine<a id="orgheadline11"></a>

Engine是总的调用接口。

```cpp
void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority) {
    ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
    OprBlock* opr_block = OprBlock::New();
    opr_block->opr = threaded_opr;

    opr_block->wait.store(static_cast<int>(
                              threaded_opr->const_vars.size() +
                              threaded_opr->mutable_vars.size() + 1));
    opr_block->ctx = exec_ctx;
    opr_block->priority = priority;
    ++pending_;
    // Add read dependencies.
    for (auto&& i : threaded_opr->const_vars) {
        i->AppendReadDependency(opr_block);
    }
    // Add write dependencies.
    for (auto&& i : threaded_opr->mutable_vars) {
        i->AppendWriteDependency(opr_block);
    }
    if (opr_block->decr_wait() == 0) {
        this->PushToExecute(opr_block, true);
    }
}
```

代码是比较清楚的，主要是 `AppendReadDependency` 和 `AppendWriteDependency` 的部
分，实际上就是把op加到它所依赖的Var的队列中, 最后检查wait是不是为0，如果为0，那
么意味着所有依赖都已经就绪，可以直接扔到执行引擎上执行了(PushToExecute),对于不同
的执行引擎, `PushToExecute` 的实现是不一样的。最终都会执行 `ExecuteOprBlock`.

```cpp
void ExecuteOprBlock(RunContext run_ctx, OprBlock *opr_block) {
    ThreadedOpr* threaded_opr = opr_block->opr;
    CallbackOnComplete callback = this->CreateCallback(
        ThreadedEngine::OnCompleteStatic, threaded_opr);
    bool debug_info = (engine_info_ && debug_push_opr_ == opr_block);
    if (!shutdown_phase_) {
      try {
        threaded_opr->fn(run_ctx, callback);
      } catch(dmlc::Error &e) {
        std::string what = e.what();
      }
    } else {
      callback();
    }

    OprBlock::Delete(opr_block);
  }
```

上述代码实际就是执行op中的函数，同时在结束的时候运行 `OnCompleteStatic`.

```cpp
void ThreadedEngine::OnCompleteStatic(
    Engine *engine, void *threaded_opr) {
  static_cast<ThreadedEngine*>(engine)->OnComplete(
      static_cast<ThreadedOpr*>(threaded_opr));
}
```

显然， `OnCompleteStatic` 就是执行 OnComplete。

```cpp
inline void ThreadedEngine::OnComplete(ThreadedOpr* threaded_opr) {
  // Mark complete for read variables
  for (auto&& i : threaded_opr->const_vars) {
    i->CompleteReadDependency([this](OprBlock* opr) {
        this->PushToExecute(opr, false);
      });
  }
  // Mark complete for write variables.
  for (auto&& i : threaded_opr->mutable_vars) {
    bool debug_info = (engine_info_ && debug_wait_var_ == i);
    if (debug_info) {
      LOG(INFO) << "Complete write dep for " << i;
    }
    bool to_delete = i->CompleteWriteDependency(
        [this, debug_info](OprBlock* opr) {
          if (debug_info) {
            LOG(INFO) << "PushToExecute " << opr;
            debug_push_opr_ = opr;
          }
          this->PushToExecute(opr, false);
          if (debug_info) {
            LOG(INFO) << "Fin PushToExecute " << opr;
          }
        });
    if (to_delete) {
      ThreadedVar::Delete(i);
    }
  }
  int npending;
  {
    std::unique_lock<std::mutex> lock{finished_m_};
    npending = --pending_;
  }
  CHECK_GE(npending, 0);
  if (npending == 0) {
    // no need to grab lock when notify.
    finished_cv_.notify_all();
  }

  // delte operator if it is temperory
  if (threaded_opr->temporary) {
    ThreadedOpr::Delete(threaded_opr);
  }
}
```

这个函数实际上就是Op完成后用来更新Var的队列的，在内部会调用每一个读依赖的
`CompleteReadDependency` 以及写依赖的 `CompleteWriteDependency`, 注意上面传递给
`CompleteReadDependency` 和 `CompleteWriteDependency` 的匿名函数(dispatcher)中主
要是调用了 `PushToExecute`.

## 总结<a id="orgheadline10"></a>

通过 `Push` 将Op的各种依赖加入相应的Var的队列，并且当依赖都满足的时候将op丢入执
行引擎执行，当执行引擎完成后，调用 `Complete` 系列的函数来更新Var的队列,在更新队
列的过程中，它又会将依赖就绪的Op丢入执行引擎执行，这样一直循环，直到所有的计算过
程都完成。
