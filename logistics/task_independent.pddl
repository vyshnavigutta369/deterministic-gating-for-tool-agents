(define (problem logistics-independent-3)
(:domain logistics)
(:objects
 apt1 apt2 apt3 - airport
 pos1 pos2 pos3 - location
 cit1 cit2 cit3 - city
 tru1 tru2 tru3 - truck
 obj1 obj2 obj3 - package)

(:init
 (in-city pos1 cit1) (in-city apt1 cit1)
 (in-city pos2 cit2) (in-city apt2 cit2)
 (in-city pos3 cit3) (in-city apt3 cit3)
 (at tru1 pos1) (at obj1 pos1)
 (at tru2 pos2) (at obj2 pos2)
 (at tru3 pos3) (at obj3 pos3))

(:goal (and (at obj1 apt1) (at obj2 apt2) (at obj3 apt3)))
)
