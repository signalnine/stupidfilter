/*=============================================================================
    Copyright (c) 2001-2006 Joel de Guzman

    Distributed under the Boost Software License, Version 1.0. (See accompanying 
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/
#if !defined(FUSION_NEXT_IMPL_06052005_0900)
#define FUSION_NEXT_IMPL_06052005_0900

#include <boost/fusion/algorithm/query/detail/find_if.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

namespace boost { namespace fusion
{
    struct filter_view_iterator_tag;

    template <typename First, typename Last, typename Pred>
    struct filter_iterator;

    namespace extension
    {
        template <typename Tag>
        struct next_impl;

        template <>
        struct next_impl<filter_view_iterator_tag>
        {
            template <typename Iterator>
            struct apply
            {
                typedef typename Iterator::first_type first_type;
                typedef typename Iterator::last_type last_type;
                typedef typename Iterator::pred_type pred_type;

                typedef typename
                    mpl::eval_if<
                        result_of::equal_to<first_type, last_type>
                      , mpl::identity<last_type>
                      , result_of::next<first_type>
                    >::type
                next_type;

                typedef typename detail::static_find_if<
                    next_type, last_type, pred_type>
                filter;

                typedef filter_iterator<
                    typename filter::type, last_type, pred_type>
                type;

                static type
                call(Iterator const& i)
                {
                    return type(filter::call(i.first));
                }
            };
        };
    }
}}

#endif


