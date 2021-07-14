#include<realm.h>
#include<stdint.h>
#include<thread>
#include<chrono>


extern "C" {
Realm::Barrier createRealmBarrier();
void destroyRealmBarrier(Realm::Barrier& b);
void realmInitRuntime(int argc, char**argv);
void realmSpawn(Realm::Barrier& b,
      Realm::Processor::TaskFuncPtr func,
		  const void* args,
		  size_t arglen
      );
void realmSync(Realm::Barrier& b);
size_t realmGetNumProcs();
  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    unsigned cur_task;
  } context;
context getRealmCTX();
}

void task(const void* args, uint64_t arglen, const void* mem, uint64_t memlen, Realm::Processor p){
  int i = *(int*)args;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  printf("hello from task %d / %d\n", i, realmGetNumProcs());
}


int main(int argc, char** argv){
  realmInitRuntime(argc, argv);
  Realm::Barrier b = createRealmBarrier();
  auto n = realmGetNumProcs();
  int a[n]; for(unsigned int i=0; i < n; i++) a[i] = i;
  for(unsigned int i=0; i<n; i++){
    realmSpawn(b, &task, (void*)(a+i), sizeof(int));
  }
  realmSync(b);
  destroyRealmBarrier(b);
}
