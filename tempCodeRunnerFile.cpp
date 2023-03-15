// #include <bits/stdc++.h>
#include<iostream>
#include<vector>
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<stack>
#include<unordered_set>
#include<cmath>
#include<numeric>
#include<iterator>
#include<set>
#include<unordered_map>
#include<map>
#include<queue>
#include<cstring>
#include<stdio.h>
#include<fstream>
#include<iomanip>
#include<functional>
#include<bitset>
#include<chrono>
#include<climits>
#include<assert.h>
// #define int double
// #define endl "\n"
#define ff first
#define ss second
#define ll long long int
#define db double
#define vi vector<int>
#define vll vector<ll>
#define vvi vector<vector<int>>
#define vvll vector<vector<ll>>
#define vb vector<bool>
#define pii pair<int,int>
#define pll pair<ll,ll>
#define vpii vector<pii>
#define vpll vector<pll>
#define rep(i,n) for(ll i=0;i<n;i++)
#define rept(i,x,n) for(ll i=x;i<n;i++)
#define pb push_back
#define rev(i,a,b) for(ll i=a;i>=b;i--)
#define all(v) v.begin(),v.end()
#define null NULL
#define node ListNode
#define tc ll tes;cin>>tes;while(tes--)
#define fast ios_base::sync_with_stdio(false);cin.tie(NULL); cout.tie(NULL);
#define coutv(v) for(auto it: (v))cout<<it<<" ";cout<<endl;
#define cout2d(v) for(auto it: (v)) {for(auto j:it) cout<<j<<" ";cout<<endl;}
#define cinv(v,n) rep(i,n){cin>>v[i];}
#define pr(s) cout<<s<<endl;
 
 
using namespace std;
 
#ifndef ONLINE_JUDGE
#define debug(x) cout<<#x<<" is "<<x<<endl;
#define debugv(v) cout<<#v<< " is :-"<<endl;for(auto it: (v))cout<<it<<" ";cout<<endl;
#define debug2d(v) cout<<#v<< " is :-"<<endl;cn=0;for(auto it: (v)) {cout<<cn<<" => ";cn++; for(auto j:it) cout<<j<<" ";cout<<endl;}
#define pline(x) cout<<x<<" line is reached"<<endl;
#define debugvp(v) cout<<#v<<" is :-"<<endl;for(auto i: v){cout<<"("<<i.ff<<","<<i.ss<<")"<<" ";}cout<<endl;
#define prline cout<<endl;
#define debugpair(p) cout<<#p<<" is ("<<p.ff<<","<<p.ss<<")"<<endl;
#define print(s) cout<<s<<endl;
#else
// #define debug(x)
// #define debugv(v)
// #define debug2d(v)
// #define pline(x)
// #define debugvp(v)
// #define prline
// #define debugpair(p)
// #define print(s) 
#endif
ll cn;

int n=26;

struct node{
    bool end;
    vector<node*> child;
    node(){
        end=0;
        rep(i,n){
            child.push_back(NULL);
        }
    }

};

void insert(node* root,string s){
    int n=s.length();
    node* temp=root;
    rep(i,n){
        if(!temp->child[s[i]-'a']){
            temp->child[s[i]-'a']=new node();
        }
        temp=temp->child[s[i]-'a'];
    }
    temp->end=1;
}

bool search(node* root, string s){
    int n=s.length();
    rep(i,n){
        if(!root->child[s[i]-'a']){
            return 0;
        }
        root=root->child[s[i]-'a'];
    }
    if(!root->end){
        return 0;
    }
    return 1;

}


int main(){
    fast
    node* root = new node();
    rep(i,10){
        int x;
        string s;
        cin>>x>>s;
        if(x==1){
            insert(root,s);
        }
        else{
            bool x=search(root,s);
            print("reached")
            cout<<x<<endl;
        }


    }
 
 
    return 0;
}